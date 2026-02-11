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
        



    def sleep(self, target_layer=1, total_steps=3000, temperature=1.0):
        """
        Sleep trains memories[target_layer] (plus a temporary linear head) to predict the
        NEXT subsampled lower-layer hidden state from the past K subsampled lower-layer states.

        Subsampling:
        delta = short_term_memory ** (target_layer - 1)
        We only take one teacher state every `delta` teacher ticks.
        The student sees a stream on that coarser clock.
        """
        if self.wake is True:
            self.wake = False

        # ----------------------------
        # Freeze everything except target memory
        # ----------------------------
        self._freeze_memories(start_layer=0)
        self._unfreeze_memory(target_layer)
        self._freeze_heads()

        K = self.short_term_memory
        H_lower  = self.hidden_sizes[target_layer - 1]
        H_target = self.hidden_sizes[target_layer]

        # TRUE subsampling stride for the lower stream that drives this layer
        delta = self.short_term_memory ** (target_layer)  # layer1=1, layer2=K, layer3=K^2, ...
        # (If you want a different definition, change it here.)

        # ----------------------------
        # Temporary linear head (sleep-only)
        # ----------------------------
        tmp_head = nn.Linear(H_target, H_lower).to(self.device)

        opt_kwargs = self.optimizer_kwargs or {}
        sleep_opt = self.optimizer_class(
            list(self.memories[target_layer].parameters()) + list(tmp_head.parameters()),
            lr=self.lr_layers,
            **opt_kwargs
        )
        loss_func = nn.MSELoss()

        # buffer on the SUBSAMPLED stream: K past + 1 next target
        buf = deque(
            [torch.zeros(1, 1, H_lower, device=self.device) for _ in range(K + 1)],
            maxlen=K + 1
        )

        # ----------------------------
        # Teacher init
        # ----------------------------
        h_teacher = None
        if target_layer == 1:
            x = torch.tensor(0, device=self.device).view(1, 1)  # token
        else:
            x = torch.zeros(1, 1, H_lower, device=self.device)  # lower vec stream

        # student carry for target memory (TBPTT)
        h_ = None
        last_loss = None

        # We interpret total_steps as TEACHER ticks.
        # Student updates happen only when we have a new subsampled state (every delta ticks).
        for tick in range(total_steps):
            # ---- teacher tick (no grad inside these methods) ----
            if target_layer == 1:
                h_teacher, x = self._teacher_step_layer0(x, h_teacher, temperature=temperature)  # (1,1,H0)
            else:
                h_teacher, x = self._teacher_step_higher(target_layer - 1, x, h_teacher)         # (1,1,H_{l-1})

            # ---- subsample: only keep every delta-th teacher state ----
            if (tick % delta) != 0:
                continue

            # push subsampled teacher state
            buf.append(h_teacher)  # detached already

            # need a full window to train
            if len(buf) < (K + 1):
                continue

            # ---- build training pair on subsampled clock ----
            past_seq = torch.cat(list(buf)[:K], dim=1)   # (1, K, H_lower)
            target_next = list(buf)[-1].detach()         # (1, 1, H_lower)

            # ---- student forward (grad-enabled) ----
            if h_ is not None:
                h_ = h_.detach()  # fixed TBPTT truncation each update (or do every K updates if you want)

            h_t, h_ = self.memories[target_layer](past_seq, h_)  # h_t: (1,1,H_target)

            # decode next lower hidden using temporary linear head
            pred_next = tmp_head(h_t.transpose(0, 1).squeeze(1))  # (B,H_lower) with B=1
            pred_next = pred_next.view(1, 1, H_lower)             # (1,1,H_lower)

            loss = loss_func(pred_next, target_next)
            last_loss = loss

            sleep_opt.zero_grad(set_to_none=True)
            loss.backward()
            sleep_opt.step()

        if last_loss is not None:
            print(f"[sleep] layer={target_layer} delta={delta} final MSE={float(last_loss.item()):.6f}")
        else:
            print(f"[sleep] layer={target_layer} delta={delta} no updates ran")


                
@torch.no_grad()
def _sample_next_token(logits, temperature=1.0):
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)  # (B,)               