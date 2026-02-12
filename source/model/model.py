import torch
import torch.nn as nn
from torch import optim
from collections import deque 
from .prediction import PredictionFiLM
from .memory import Memory
from .layer import Layer


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
        self.wake = False

        # ------------------------------------------------------------
        # 1. BUILD LAYERS (with correct context sizes)
        # ------------------------------------------------------------
        self.layers = nn.ModuleList()

        for l in range(self.total_layers):
            if l == 0:
                loss_function = nn.CrossEntropyLoss()
            else:
                loss_function = nn.MSELoss()

            input_size = self.vocab_size if l == 0 else self.hidden_sizes[l-1]
            self.layers.append(
                Layer(
                        input_size,
                        self.hidden_sizes[l],
                        loss_function,
                        optimizer_class=self.optimizer_class,    
                        optimizer_kwargs=self.optimizer_kwargs,               
                        embedding_dim=self.embedding_dim_l0 if l==0 else None,
                        layer=l,
                        context_size=self.hidden_sizes[l] if l+1<self.total_layers else 0,
                )
            )
        

        # ------------------------------------------------------------
        # 2. STATE 
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

    def _freeze_all(self):
        for l in range(self.total_layers):
            self.layers[l].freeze()


    def wake_step(self, x, y, h_=None):
        """
        """
        if self.wake is False:
            self.step = 0
            self._freeze_all()
            self.layers[0].unfreeze()
            self.wake = True

            for l in range(self.total_layers):
                H = self.hidden_sizes[l]
                self.h_states[l] = torch.zeros(1, 1, H, device=self.device)
                

        self.step += 1
        t = self.step

        x = x.to(self.device)
        y = y.view(-1).long().to(self.device)

        
        # ------------------------------------------------
        # Bottom-up memory updates
        # ------------------------------------------------
        # Layer 0 (trainable)
        h0, h_ = self.layers[0].memory(x, h_)
        self.h_states[0] = h0.detach()

        # Upper layers (frozen weights, state only)
        with torch.no_grad():
            for l in range(1, self.total_layers):
                stride = self.short_term_memory ** l
                if t % stride != 0:
                    continue

                x_vec = self.h_states[l-1].transpose(0, 1)  # (B,1,H_{l-1})
                self.h_states[l] = self.layers[l].memory.encode_step_from_vec(
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
                logits = self.layers[0].prediction(h0, context=context)  # (B,1,V)
            else:
                # produce context for lower layer
                context = self.layers[l].prediction(z, context=context)  # (B,1,H_{l-1})

        # logits = logits.squeeze(1)  # (B,V)

        # ------------------------------------------------
        # Global loss (ONLY one)
        # ------------------------------------------------
        loss = self.layers[0].compute_pred_loss(logits, y)
        
        self.layers[0].optimizer.zero_grad()
        loss.backward()
        self.layers[0].optimizer.step()

        return logits.detach(), loss.item(), h_.detach()
    
    def sleep(self, target_layer, total_steps):

        self.wake = False
        self.layers[target_layer].unfreeze()

        buf = deque(
            [torch.zeros(1, 1, self.hidden_sizes[target_layer-1], device=self.device) for _ in range(self.short_term_memory + 1)],
            maxlen=self.short_term_memory + 1
        )
        x, h, h_pass = None, None, None
        for ii in range(1,total_steps):
            x, h = self.layers[target_layer-1].generate_sample(x, h, temperature=10.0)

            if ii % self.short_term_memory == 0:

                logits, _, h_pass = self.layers[target_layer](torch.cat(list(buf), dim=1), h_pass)
                h_pass = h_pass.detach()

                loss = self.layers[target_layer].compute_pred_loss(logits, h)

                self.layers[target_layer].optimizer.zero_grad()
                loss.backward()
                self.layers[target_layer].optimizer.step()

                buf.append(
                    h.detach()
                )

            if ii%1000==0:
                print("Sleep loss ", loss.item())

        self.layers[target_layer].freeze()




    

    