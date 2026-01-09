import torch
import torch.nn as nn
from torch import optim
from collections import deque
from .layer import Layer  
from ..utils.loss import CrossEntropyLayerLoss, MSELayerLoss, MaskedMSELoss


class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        defaults = dict(
            total_layers = 3,
            vocab_size = None,
            hidden_sizes = None,
            embedding_dim_l0 = None,
            short_term_memory = 3,
            tau = 0.5,
            threshold = 1e-4,
            lr_layers = None,
            optimizer_class = optim.Adam,
            optimizer_kwargs = None,
            sleep_steps = None,
            device = "cpu",
        )
        for k, v in {**defaults, **kwargs}.items():
            setattr(self, k, v)

        self.device = torch.device(self.device)
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {"lr": 1e-3, "weight_decay": 1e-8}

        assert self.vocab_size is not None
        assert self.hidden_sizes is not None
        assert self.embedding_dim_l0 is not None
        assert self.lr_layers is not None
        assert len(self.hidden_sizes) == self.total_layers
        assert len(self.lr_layers) == self.total_layers

        if self.sleep_steps is None:
            self.sleep_steps = {l: 100 for l in range(1, self.total_layers)}

        self.step = 0
        # ------------------------------------------------------------
        # 4. BUILD LAYERS (with correct context sizes)
        # ------------------------------------------------------------
        self.layers = nn.ModuleList()

        for l in range(self.total_layers):

            if l == 0:
                inp = self.vocab_size
                emb = self.embedding_dim_l0
            else:
                inp = self.hidden_sizes[l-1]
                emb = None

            # context from layer l+1:
            #   - prediction dimension  = hidden_sizes[l]
            #   - memory z dim      = hidden_sizes[l+1]
            if l + 1 < self.total_layers:
                ctx = self.hidden_sizes[l] + self.hidden_sizes[l+1]
            else:
                ctx = 0

            if l == 0:
                loss_fn = nn.CrossEntropyLoss()
            else:
                loss_fn = MaskedMSELoss()

            self.layers.append(
                Layer(
                    input_size   = inp,
                    hidden_size  = self.hidden_sizes[l],
                    loss_function= loss_fn,
                    optimizer_class = self.optimizer_class,   # unused in wake
                    optimizer_kwargs= self.optimizer_kwargs,  # "
                    embedding_dim= emb,
                    layer       = l,
                    context_size= ctx,
                    tau         = self.tau,
                ).to(self.device)
            )

        # ------------------------------------------------------------
        # 5. WAKE OPTIMIZER: L0 memory + prediction heads of all layers
        # ------------------------------------------------------------
        param_groups = []

        # L0 memory + its prediction head use lr_layers[0]
        pg0 = {
            "params": list(self.layers[0].memory.parameters()) +
                      list(self.layers[0].prediction.parameters()),
            "lr": self.lr_layers[0],
        }
        param_groups.append(pg0)

        # prediction heads for upper layers, each with its own lr_layers[l]
        for l in range(1, self.total_layers):
            pg = {
                "params": list(self.layers[l].prediction.parameters()),
                "lr": self.lr_layers[l],
            }
            param_groups.append(pg)

        # copy other optimizer kwargs except 'lr'
        opt_kwargs = {k: v for k, v in self.optimizer_kwargs.items() if k != "lr"}
        self.optimizer_wake = self.optimizer_class(param_groups, **opt_kwargs)

        # ------------------------------------------------------------
        # 6. STATE / EMA BUFFERS
        # ------------------------------------------------------------
        self.h_states = {}
        self.last_pred = {}
        self.h        = {}

        for l in range(self.total_layers):
            H = self.hidden_sizes[l]
            self.h_states[l] = torch.zeros(1, 1, H, device=self.device)

            if l == 0:
                self.last_pred[l] = torch.zeros(1, 1, self.vocab_size,
                                                device=self.device)
            else:
                # initial prediction from zero z_states
                self.last_pred[l] = self.layers[l].prediction(self.h_states[l], None)



    # ===================================================================
    def summary(self):
        print("\n===== Model Summary =====")
        print(f"Total layers: {self.total_layers}")
        print(f"Hidden sizes: {self.hidden_sizes}")
        print(f"Sleep steps: {self.sleep_steps}")
        print(f"Device: {self.device}")
        print("=================================\n")

    '''def wake_step(self, x, y):
        """
        Joint wake update:
        - L0 memory + prediction are trained every step.
        - Upper-layer prediction heads are trained when their horizon closes.
        - Context to each layer l is [pred_{l+1}, z_states_{l+1}] (when l+1 exists).
        """

        self.step += 1
        t = self.step

        x = x.to(self.device)
        y = y.to(self.device)

        # ---------------- LAYER 0 FORWARD ----------------
        logits_rec0, z0, _ = self.layers[0].memory(x, None)

        if self.total_layers > 1:
            # pred_1 has dim H0, z_states[1] has dim H1
            ctx_pred1 = self.last_pred[1].detach()
            ctx_mem1  = self.z_states[1].detach()
            #print(ctx_pred1.shape, ctx_mem1.shape)
            ctx0      = torch.cat([ctx_pred1, ctx_mem1], dim=-1)
        else:
            ctx0 = None

        pred0 = self.layers[0].prediction(z0, ctx0)

        loss_mem0  = self.layers[0].compute_mem_loss(logits_rec0, x)
        loss_pred0 = self.layers[0].compute_pred_loss(pred0, y)

        if loss_mem0>1e-3:
            total_loss = loss_mem0 + loss_pred0
        else:
            total_loss = loss_pred0


        # update L0 EMA/state (no grad restriction)
        self.z_states[0]= z0.unsqueeze(1)
        self.last_pred[0]= pred0

        # ------------- UPPER-LAYER PREDICTION LOSSES -------------
        for l in range(1, self.total_layers):
            stride = self.short_term_memory ** l
            if t % stride != 0:
                continue

            target_z = self.z_states[l - 1].detach()   # dim H_{l-1}
            pred     = self.last_pred[l]            # same dim
            loss_l   = self.layers[l].compute_pred_loss(pred, target_z)
            total_loss = total_loss + loss_l
        
        # ------------- JOINT BACKWARD + STEP -------------
        self.optimizer_wake.zero_grad()
        total_loss.backward()
        self.optimizer_wake.step()

        # ------------- UPDATE UPPER STATES & NEW PREDICTIONS -------------
        for l in range(1, self.total_layers):
            stride = self.short_term_memory ** l
            if t % stride != 0:
                continue

            current_z = self.z_states[l - 1].detach()

            # context from ABOVE: [pred_{l+1}, z_states_{l+1}]
            if l + 1 < self.total_layers:
                ctx_pred = self.last_pred[l + 1].detach()   # dim H_l
                ctx_mem  = self.z_states[l + 1].detach()       # dim H_{l+1}
                ctx_l    = torch.cat([ctx_pred, ctx_mem], dim=-1)
            else:
                ctx_l = None

            z_l, self.h[l] = self.layers[l].memory.encode_step_from_vec(
                current_z, self.h[l]
            )
            z_l = z_l.detach()      # freeze dynamics

            pred_l = self.layers[l].prediction(z_l, ctx_l)
            self.last_pred[l] = pred_l

            # update state for layer l
            self.z_states[l]   = z_l

        return dict(
            step=t,
            loss_mem=loss_mem0.item(),
            loss_pred=loss_pred0.item(),
            logits_rec0=logits_rec0.detach(),
            logits_pred0=pred0.detach(),
        )'''
    
    def wake_step(self, x, y):
        """
        Joint wake update:
        - L0 memory + prediction are trained every step.
        - Upper-layer prediction heads are trained when their horizon closes.
        - Context to each layer l is [pred_{l+1}, z_states_{l+1}] (when l+1 exists).
        """

        self.step += 1
        t = self.step

        x = x.to(self.device)
        y = y.to(self.device)

        # ---------------- LAYER 0 FORWARD ----------------
        logits_rec0, z0 = self.layers[0].memory(x, None)
        self.z_states[0]   = z0.unsqueeze(1)
        

        for l in range(1, self.total_layers):
            stride = self.short_term_memory ** l
            if t % stride != 0:
                continue

            current_z = self.z_states[l - 1]
            self.z_states[l], self.h[l] = self.layers[l].memory.encode_step_from_vec(
                                    current_z, self.h[l]
                                )

        for l in reversed(range(self.total_layers)):
            # context from ABOVE: [pred_{l+1}, z_states_{l+1}]
            if l + 1 < self.total_layers:
                ctx_pred = self.last_pred[l + 1]  # dim H_l
                ctx_mem  = self.z_states[l + 1]      # dim H_{l+1}
                ctx_l    = torch.cat([ctx_pred, ctx_mem], dim=-1)
            else:
                ctx_l = None

            pred_l = self.layers[l].prediction(self.z_states[l], ctx_l)
            self.last_pred[l] = pred_l

        loss_mem0  = self.layers[0].compute_mem_loss(logits_rec0, x)
        print(self.last_pred[0][0][0][0],y[0])
        loss_pred0 = self.layers[0].compute_pred_loss(self.last_pred[0][0][0][0], y[0][0])

        if loss_mem0>self.threshold:
            total_loss = loss_mem0 + loss_pred0
        else:
            total_loss = loss_pred0
        
        
        # ------------- JOINT BACKWARD + STEP -------------
        self.optimizer_wake.zero_grad()
        total_loss.backward()
        self.optimizer_wake.step()

        
        return dict(
            step=t,
            loss_mem=loss_mem0.item(),
            loss_pred=loss_pred0.item(),
            logits_rec0=logits_rec0.detach(),
            logits_pred0=self.last_pred[0].detach(),
        )
    

    def sleep_train_layers(
            self
    ):
        self.step = 0 #reset wake step size 

        input_buffer = {}
        z_states = {}

        for layer in range(self.total_layers-1):
            H_size = self.layers[layer].hidden_size
            input_buffer[layer] = deque(
                [torch.zeros(1, 1, H_size, device=self.device) for _ in range(self.short_term_memory)],
                maxlen=self.short_term_memory
            )
        
        # Training loop
        total_steps = self.sleep_steps
        for t in range(1,total_steps+1):
            if t == 1:
                x_next, z, h = self.layers[0].generate_sample()
            else:
                x_next, z, h = self.layers[0].generate_sample(x=x_next, h0=h)

            z_states[0] = z.clone()

            for layer in range(1,self.total_layers):
                if t%self.short_term_memory**layer == 0:
                    input_buffer[layer-1].append(z_states[layer-1].clone())

                    window = torch.cat(list(input_buffer[layer-1]), dim=1)  # (1, stm, H_lower)
                    loss, _, _, _ = self.layers[layer].train_memory(
                                            window, threshold=0.01
                                        )
                    
                    with torch.no_grad():
                        _, z_, _ = self.layers[layer].memory(window)
                        z_states[layer] = z_.unsqueeze(1)

            



    # =========================
    # Sleep replay (for a layer-pair)
    # =========================
    def sleep_train_layer(
        self, target_layer
    ):
        r"""
            Sleep-phase replay for hierarchical consolidation in the model.

            This function trains ONLY the specified `target_layer` using replayed
            latent trajectories generated from the *lower* layer (target_layer - 1),
            analogous to hippocampus-cortex consolidation in the brain.

            ─────────────────────────────────────────────────────────────
            BIOLOGICAL ANALOGY
            ─────────────────────────────────────────────────────────────
            - Source layer (L-1): acts like the hippocampus → generates replayed activity.
            - Target layer (L): acts like cortex → slowly learns stable representations.
            - Replay z-trajectories: hippocampal reactivation during NREM sleep.
            - No gradient flows to lower layers (hippocampus is a teacher, not a student).
            - Each layer has its own local optimizer (no global backprop).
            
            ─────────────────────────────────────────────────────────────
            WHAT THE FUNCTION DOES
            ─────────────────────────────────────────────────────────────
            1. Select source layer S = target_layer - 1.
            2. Generate synthetic replay from S using:
                x_next, z_S, h_S = generate_sample(...)
            where z_S is the variational latent code sampled from q(z|x).
            3. Smooth the replayed z_S over time using exponential moving average (EMA).
            4. Store a sliding window of the last `short_term_memory` z vectors.
            5. Every `short_term_memory` steps, train the target layer on:
                input  = stacked window
                target = most recent z
            6. Because each layer has its own optimizer, only the target layer learns.
            Lower layers remain frozen automatically.

            ─────────────────────────────────────────────────────────────
            ARGUMENTS
            ─────────────────────────────────────────────────────────────
            target_layer : int
                Index of the layer to train during sleep.
                The source layer is (target_layer - 1).

            
            ─────────────────────────────────────────────────────────────
            RETURNS
            ─────────────────────────────────────────────────────────────
            None
                Prints final sleep replay loss for the target layer.

            ─────────────────────────────────────────────────────────────
            KEY PROPERTIES OF THIS IMPLEMENTATION
            ─────────────────────────────────────────────────────────────
            - Uses z (sampled variational latent) as replay representation.
            - Replay is context-free → matches hippocampal spontaneous replay.
            - Lower layers never receive gradients (natural freezing).
            - Upper layers learn stable long-timescale patterns.
            - Windowing produces temporal receptive fields (place-field-like tuning).
        """
        self.step = 0 #reset wake step size

        source_layer = target_layer - 1


        # Initialize buffers
        H_lower = self.layers[source_layer].hidden_size
        stm_queue = deque(
            [torch.zeros(1, 1, H_lower, device=self.device) for _ in range(self.short_term_memory)],
            maxlen=self.short_term_memory
        )

        train_stride = self.short_term_memory

        # Training loop
        total_steps = self.sleep_steps * train_stride
        for t in range(total_steps):
            if t == 0:
                x_next, z, h = self.layers[source_layer].generate_sample()
            else:
                x_next, z, h = self.layers[source_layer].generate_sample(x=x_next, h0=h)

            
            # --- Downsampling / training trigger ---
            if t % train_stride == 0:
                stm_queue.append(z.clone())
                window = torch.cat(list(stm_queue), dim=1)  # (1, stm, H_lower)
                loss, _, _, _ = self.layers[target_layer].train_memory(
                                        window, threshold=0.1
                                    )


        print('Sleeping memory loss ', loss)
