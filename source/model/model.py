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
            vocab_size = None,
            hidden_sizes = None,
            embedding_dim_l0 = None,
            short_term_memory = 3,
            lr_layers = 1e-3,
            optimizer_class = optim.Adam,
            optimizer_kwargs = None,
            sleep_steps = None,
            device = "cpu",
        )
        for k, v in {**defaults, **kwargs}.items():
            setattr(self, k, v)

        self.device = torch.device(self.device)
        
        assert self.vocab_size is not None
        assert self.hidden_sizes is not None
        assert self.embedding_dim_l0 is not None
        assert self.lr_layers is not None
        assert len(self.hidden_sizes) == self.total_layers

        if self.sleep_steps is None:
            self.sleep_steps = {l: 100 for l in range(1, self.total_layers)}

        self.step = 1
        # ------------------------------------------------------------
        # 1. BUILD LAYERS (with correct context sizes)
        # ------------------------------------------------------------
        self.memories = nn.ModuleList()
        self.heads = nn.ModuleList()

        for l in range(self.total_layers):
            output_size = self.hidden_sizes[l-1] if l>0 else self.vocab_size

            self.heads.append(
                PredictionFiLM(
                    self.hidden_sizes[l],
                    output_size,
                    context_size=self.hidden_sizes[l] if l+1<self.total_layers else 0
                )
            )

            input_size = self.vocab_size if l == 0 else self.hidden_sizes[l-1]
            self.memories.append(
                Memory(
                    input_size=input_size,
                    hidden_size=self.hidden_sizes[l],
                    embedding_dim=self.embedding_dim_l0,
                    layer=l
                )
            )

        # ------------------------------------------------------------
        # 2. Define optmizers and loss function 
        # ------------------------------------------------------------
        self.wake = False

        params = []
        # train layer0 memory
        params += list(self.memories[0].parameters())
        # train all heads
        for head in self.heads:
            params += list(head.parameters())

        opt_kwargs = self.optimizer_kwargs 
        self.wake_opt = self.optimizer_class(params, lr=self.lr_layers, **opt_kwargs)


        # ------------------------------------------------------------
        # 3. STATE 
        # ------------------------------------------------------------
        self.h_states = {}

        for l in range(self.total_layers):
            H = self.hidden_sizes[l]
            self.h_states[l] = torch.zeros(1, 1, H, device=self.device)
            


    # ===================================================================
    def summary(self):
        print("\n===== Model Summary =====")
        print(f"Total layers: {self.total_layers}")
        print(f"Hidden sizes: {self.hidden_sizes}")
        print(f"Sleep steps: {self.sleep_steps}")
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
                self.h_states[l] = torch.zeros(1, 1, H, device=self.device)
                
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
        h0, h_ = self.memories[0](x, h_)
        self.h_states[0] = h0

        # Upper layers (frozen weights, state only)
        with torch.no_grad():
            for l in range(1, self.total_layers):
                stride = self.short_term_memory ** l
                if t % stride != 0:
                    continue

                x_vec = self.h_states[l-1].transpose(0, 1)  # (B,1,H_{l-1})
                self.h_states[l] = self.memories[l].encode_step_from_vec(
                    x_vec, self.h_states[l]
                )
        # ------------------------------------------------
        # Top-down context construction via heads
        # ------------------------------------------------
        context = None
        for l in reversed(range(self.total_layers)):
            z = self.h_states[l].transpose(0, 1)  # (B,1,H_l)

            if l == 0:
                # final prediction head
                logits = self.heads[0](z, context=context)  # (B,1,V)
            else:
                # produce context for lower layer
                context = self.heads[l](z, context=context)  # (B,1,H_{l-1})

        logits = logits.squeeze(1)  # (B,V)

        # ------------------------------------------------
        # Global loss (ONLY one)
        # ------------------------------------------------
        loss = nn.functional.cross_entropy(logits, y)

        self.wake_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.wake_opt.step()

        return logits.detach(), loss.item(), h_.detach()



    @torch.no_grad()
    def _teacher_step_layer0(self, token, h0_carry, temperature=1.0):
        """
        One teacher tick for layer0: token -> memory0 -> logits (context-free) -> next token.
        Returns: h0 (1,B,H0) detached, next_token (B,), updated carry
        """
        B = token.size(0)
        x = token.view(B, 1)  # (B,1)
        h0_carry = self.memories[0].encode_step_from_token(x, h0_carry)  

        z0 = h0_carry.transpose(0, 1)  # (B,1,H0)
        logits = self.heads[0](z0, context=None).squeeze(1)  # (B,V)
        next_token = _sample_next_token(logits, temperature=temperature)

        return h0_carry, next_token

    @torch.no_grad()
    def _teacher_step_higher(self, teacher_layer, x_vec, h_teacher):
        """
        Teacher tick for layer>=1:
        input vec (B,1,H_{teacher-1}) -> memory[teacher_layer] -> h_teacher (1,B,H_teacher)
        then generate next input vec using frozen head[teacher_layer] context-free:
            x_vec_next = head[teacher_layer](h_teacher)  (B,1,H_{teacher-1})
        Returns: h_teacher_detached (1,B,H_teacher), x_vec_next_detached (B,1,H_{teacher-1}), h_teacher
        """
        h_teacher = self.memories[teacher_layer].encode_step_from_vec(x_vec, h_teacher)  # (1,B,Ht)
        z = h_teacher.transpose(0, 1)  # (B,1,Ht)
        x_vec_next = self.heads[teacher_layer](z, context=None)  # (B,1,H_{t-1})
        return h_teacher, x_vec_next
        



    def sleep(self, target_layer=1, total_steps=3000):
        if self.wake is True:
            self.wake = False

        self._freeze_memories(start_layer=0)
        self._unfreeze_memory(target_layer)
        self._freeze_heads()
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
        
        h = None
        h_ = None
        if target_layer == 1:
            x = torch.tensor(0).view(1,1)
        else:
            x = torch.zeros(1,1,self.hidden_sizes[target_layer-1])

        for ii in range(total_steps):
            if target_layer==1:
                h, x = self._teacher_step_layer0(x, h)
            else:
                h, x = self._teacher_step_higher(target_layer-1,x,h)
            
            if ii%self.short_term_memory != 0:
                continue
            
            input = torch.cat(list(input_buffer), dim=1)

            h0, h_ = self.memories[target_layer](input, h_)
            self.h_states[target_layer] = h0

            # Upper layers (frozen weights, state only)
            with torch.no_grad():
                for l in range(target_layer+1, self.total_layers):
                    stride = self.short_term_memory ** l
                    if ii % stride != 0:
                        continue

                    x_vec = self.h_states[l-1].transpose(0, 1)  # (B,1,H_{l-1})
                    self.h_states[l] = self.memories[l].encode_step_from_vec(
                        x_vec, self.h_states[l]
                    )
            # ------------------------------------------------
            # Top-down context construction via heads
            # ------------------------------------------------
            context = None
            for l in reversed(range(target_layer, self.total_layers)):
                z = self.h_states[l].transpose(0, 1)  # (B,1,H_l)

                
                context = self.heads[l](z, context=context)  # (B,1,H_{l-1})

            #logits = context.squeeze(1)  # (B,V)

            # ------------------------------------------------
            # Global loss (ONLY one)
            # ------------------------------------------------
            loss = loss_func(context, h)

            self.sleep_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.sleep_opt.step()

            input_buffer.append(h)

        print("Final MSE loss ", loss.item())

                
@torch.no_grad()
def _sample_next_token(logits, temperature=1.0):
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)  # (B,)               