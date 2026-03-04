import torch
import torch.nn as nn
from torch import optim
from collections import deque 
from .prediction import PredictionFiLM
from .memory import Memory


class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        defaults = dict(
            total_layers = 3,
            num_layers_prediction_head = 1,
            vocab_size = None,
            hidden_sizes = None,
            prediction_sizes = None,
            embedding_dim = None,
            short_term_memory = 3,
            lr_layers = 1e-3,
            recon_threshold = 1e-3,
            optimizer_class = optim.Adam,
            optimizer_kwargs = None,
            context_tag_buffer_size=10,
            device = "cpu",
        )
        for k, v in {**defaults, **kwargs}.items():
            setattr(self, k, v)

        self.device = torch.device(self.device)
        
        assert self.vocab_size is not None
        assert self.hidden_sizes is not None
        assert self.embedding_dim is not None
        assert self.lr_layers is not None
        assert len(self.hidden_sizes) == self.total_layers

        if self.prediction_sizes is None:
            self.prediction_sizes = self.hidden_sizes


        self.step = 1
        # ------------------------------------------------------------
        # 1. BUILD LAYERS (with correct context sizes)
        # ------------------------------------------------------------
        self.memories = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.context_tags = deque(
            maxlen=self.context_tag_buffer_size
        )
        self.recon_loss_ema = 0.0
        self.sleeping = True
        self.store_tags = False

        for l in range(self.total_layers):
            output_size = self.hidden_sizes[l-1] if l>0 else self.vocab_size

            self.heads.append(
                PredictionFiLM(
                    self.prediction_sizes[l],
                    output_size,
                    num_layers=self.num_layers_prediction_head,
                    context_size=self.hidden_sizes[l] if l+1<self.total_layers else 0
                )
            )

            input_size = self.vocab_size if l == 0 else self.hidden_sizes[l-1]
            self.memories.append(
                Memory(
                    input_size=input_size,
                    hidden_size=self.hidden_sizes[l],
                    embedding_dim=self.embedding_dim,
                    layer=l
                )
            )

        # ------------------------------------------------------------
        # 2. Define optmizers and loss function 
        # ------------------------------------------------------------
        self.wake = False

        params = []
        # train all heads
        for head in self.heads:
            params += list(head.parameters())

        opt_kwargs = self.optimizer_kwargs 
        self.head_wake_opt = self.optimizer_class(params, lr=self.lr_layers, **opt_kwargs)
        self.memory_wake_opt = self.optimizer_class(self.memories[0].parameters(), lr=self.lr_layers, **opt_kwargs)


        # ------------------------------------------------------------
        # 3. STATE 
        # ------------------------------------------------------------
        self.h_states = {}

        for l in range(self.total_layers):
            H = self.hidden_sizes[l]
            self.h_states[l] = torch.zeros(1, H, device=self.device)
            

    def reset_model(self):
        self.wake = False
        self.store_tags = False
        self.step = 1



    # ===================================================================
    def summary(self):
        print("\n===== Model Summary =====")
        print(f"Total layers: {self.total_layers}")
        print(f"Hidden sizes: {self.hidden_sizes}")
        print(f"Reconstruction Threshold: {self.recon_threshold}")
        print(f"Device: {self.device}")
        print("=================================\n")

    def _freeze_memories(self, start_layer: int = 0):
        for l in range(start_layer, self.total_layers):
            for p in self.memories[l].parameters():
                p.requires_grad_(False)

    def _unfreeze_memory(self, layer: int = 0):
        for p in self.memories[layer].parameters():
            p.requires_grad_(True)

    def _freeze_heads(self):
        for l in range(self.total_layers):
            for p in self.heads[l].parameters():
                p.requires_grad_(False)
    
    def _unfreeze_heads(self):
        for l in range(self.total_layers):
            for p in self.heads[l].parameters():
                p.requires_grad_(True)


    def wake_step(self, x, y, h_=None):
        """
        """
        if self.wake is False:
            self.step = 0

            for l in range(self.total_layers):
                H = self.hidden_sizes[l]
                self.h_states[l] = torch.zeros(1, H, device=self.device)
                
            self._freeze_memories(start_layer=0)
            self._unfreeze_memory(layer=0)
            self._unfreeze_heads()
            self.wake = True


        self.step += 1
        t = self.step

        x = x.to(self.device)
        y = y.view(-1).long().to(self.device)

        
        # ------------------------------------------------
        # Bottom-up memory updates
        # ------------------------------------------------
        # Layer 0 (trainable)
        recon_logit, h0, h_ = self.memories[0](x, h_)
        B, T, V = recon_logit.shape
        recon_loss = nn.functional.cross_entropy(recon_logit.reshape(B*T, V), x.reshape(B*T))
        self.recon_loss_ema = 0.1*recon_loss.item() + 0.9*self.recon_loss_ema

        if self.recon_loss_ema > self.recon_threshold:
            self.memory_wake_opt.zero_grad(set_to_none=True)
            recon_loss.backward()
            self.memory_wake_opt.step()
            self.sleeping = True
            self.store_tags = True
            

        # Upper layers (frozen weights, state only)
        with torch.no_grad():
            for l in range(self.total_layers):
                stride = self.short_term_memory ** l
                if t % stride != 0:
                    continue
                
                if l==0:
                    self.h_states[l] = self.memories[l].encode_step_from_token(
                        x[:,-1], self.h_states[l].unsqueeze(0)
                    ).squeeze(0)
                else:
                    self.h_states[l] = self.memories[l].encode_step_from_vec(
                        self.h_states[l-1], self.h_states[l]
                    )
        # ------------------------------------------------
        # Top-down context construction via heads
        # ------------------------------------------------
        context = None
        for l in reversed(range(self.total_layers)):
            if l == 0:
                # final prediction head
                if self.store_tags:
                    self.context_tags.append(
                        (self.h_states[0], context.detach())
                    )
                    self.store_tags = False
                    
                logits = self.heads[0](self.h_states[0], context=context) 
            else:
                # produce context for lower layer
                context = self.heads[l](self.h_states[l], context=context)  # (B,1,H_{l-1})

        
        logits = logits.squeeze(1)  # (B,V)
        
        
        # ------------------------------------------------
        # Global loss (ONLY one)
        # ------------------------------------------------
        
        pred_loss = nn.functional.cross_entropy(logits, y)

        if pred_loss.item() > 1e-4:
            self.head_wake_opt.zero_grad(set_to_none=True)
            pred_loss.backward()
            self.head_wake_opt.step()

        return logits.detach(), pred_loss.item(), recon_loss.item(), h_.detach()



    @torch.no_grad()
    def _teacher_step_layer0(self, h0_carry, context=None, temperature=1.0):
        """
        One teacher tick for layer0: token -> memory0 -> logits (context-free) -> next token.
        Returns: h0 (1,B,H0) detached, next_token (B,), updated carry
        """ 

        z0 = h0_carry.transpose(0, 1)  # (B,1,H0)
        logits = self.heads[0](z0, context=context).squeeze(1)  # (B,V)
        next_token = sample_topk(logits, k=self.vocab_size)  

        #_sample_next_token(logits, temperature=temperature)
        h0_carry = self.memories[0].encode_step_from_token(next_token, h0_carry) 

        return h0_carry, next_token


    def sleep(self, total_steps=100):
        if self.wake is True:
            self.wake = False

        if self.sleeping is True:
            self.sleeping = False
        else:
            return

        for target_layer in range(1, self.total_layers):
            self._freeze_memories(start_layer=0)
            self._unfreeze_memory(target_layer)
            self._freeze_heads()

            decoder_loss_ema = 0.0
            opt_kwargs = self.optimizer_kwargs 
            self.sleep_opt = self.optimizer_class(
                    self.memories[target_layer].parameters(), lr=self.lr_layers, **opt_kwargs
                )
            loss_func = nn.MSELoss()

            H_lower = self.hidden_sizes[target_layer-1]
            input_buffer = deque(
                [torch.zeros(1, 1, H_lower, device=self.device) for _ in range(self.short_term_memory)],
                maxlen=self.short_term_memory
            )
            
            for jj in range(len(self.context_tags)):
                h_states = {}
                h_states[0] = self.context_tags[jj][0].unsqueeze(0)
                h_ = None
                for layer in range(1, target_layer):
                    h_states[layer] = None 

                for ii in range(total_steps):
                    h_states[0], _ = self._teacher_step_layer0(
                            h_states[0], 
                            context=self.context_tags[jj][1]
                        )

                    for layer in range(1, target_layer):
                        if ii%self.short_term_memory**layer != 0:
                            continue

                        h_states[layer] = self.memories[layer].encode_step_from_vec(
                            h_states[layer-1], h_states[layer]
                        )
                    
                    
                    if ii%self.short_term_memory**target_layer != 0:
                        continue
                    
                    input_buffer.append(h_states[target_layer-1])
                    input = torch.cat(list(input_buffer), dim=1)

                    recon_logit, _, h_ = self.memories[target_layer](add_gaussian_noise(input), h_)
                    h_ = h_.detach()

                    recon_loss = loss_func(recon_logit, input)
                    decoder_loss_ema = 0.1*recon_loss.item() + 0.9*decoder_loss_ema

                    # if decoder_loss_ema < 1e-2 and self.memories[target_layer].decoder_is_frozen is False:
                    #     self.memories[target_layer].freeze_decoder()
                    #     print("Decoder frozen")
                    
                    # if decoder_loss_ema > .1 and self.memories[target_layer].decoder_is_frozen is True:
                    #     self.memories[target_layer].unfreeze_decoder()
                    #     print("Decoder unfrozen")

                    if decoder_loss_ema > self.recon_threshold:
                        self.sleep_opt.zero_grad(set_to_none=True)
                        recon_loss.backward()
                        self.sleep_opt.step() 

                print("Layer ", target_layer, " Sleep Loss ",jj,": ", recon_loss.item())   
        self.context_tags.clear()

    @torch.no_grad()
    def eval_step_no_train(self, x, y, h_=None):
        """
        Evaluate one step without ANY training:
        - updates states (no grad)
        - computes prediction logits and CE loss
        - computes reconstruction loss (for reporting) but does NOT backprop
        Returns: logits, pred_loss, recon_loss, h_
        """
        self.eval()

        # make sure nothing accumulates grads anywhere
        # (not strictly necessary under no_grad, but helps catch mistakes)
        for p in self.parameters():
            p.requires_grad_(False)

        x = x.to(self.device)
        y = y.view(-1).long().to(self.device)

        # -----------------------------
        # Layer 0 reconstruction forward
        # -----------------------------
        recon_logit, h0, h_pass = self.memories[0](x, h_)
        B, T, V = recon_logit.shape
        recon_loss = torch.nn.functional.cross_entropy(recon_logit.reshape(B*T, V), x.reshape(B*T))
        self.recon_loss_ema = 0.1*recon_loss.item() + 0.9*self.recon_loss_ema

        # -----------------------------
        # Bottom-up state updates (same as wake, but no grad)
        # -----------------------------
        # You used self.step to gate stride. For eval, increment too.
        self.step += 1
        t = self.step

        for l in range(self.total_layers):
            stride = self.short_term_memory ** l
            if t % stride != 0:
                continue

            if l == 0:
                self.h_states[l] = self.memories[l].encode_step_from_token(
                        x[:,-1], self.h_states[l].unsqueeze(0)
                ).squeeze(0)
            else:
                self.h_states[l] = self.memories[l].encode_step_from_vec(
                    self.h_states[l-1], self.h_states[l]
                )

        # -----------------------------
        # Top-down context via heads
        # -----------------------------
        context = None
        for l in reversed(range(self.total_layers)):
            if l == 0:
                logits = self.heads[0](self.h_states[0], context=context)
            else:
                context = self.heads[l](self.h_states[l], context=context)

        logits = logits.squeeze(1)               # (B, V)
        pred_loss = torch.nn.functional.cross_entropy(logits, y)   # scalar

        return logits, pred_loss.item(), recon_loss.item(), h_pass          
            
                
@torch.no_grad()
def sample_topk(logits, k=7, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    v, ix = torch.topk(logits, k, dim=-1)          # (B,k)
    probs = torch.softmax(v, dim=-1)
    choice = torch.multinomial(probs, 1)           # (B,1)
    return ix.gather(-1, choice).squeeze(-1)       # (B,)

def add_gaussian_noise(x, std=0.1):
    noise = torch.randn_like(x) * std
    return x + noise

