import torch
import torch.nn as nn
from torch import optim
from collections import deque
from .layer import Layer  
from ..utils.loss import CrossEntropyLayerLoss, MSELayerLoss


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
            sleep_interval = 1000,
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
            #   - memory z_ema dim      = hidden_sizes[l+1]
            if l + 1 < self.total_layers:
                ctx = self.hidden_sizes[l] + self.hidden_sizes[l+1]
            else:
                ctx = 0

            if l == 0:
                loss_fn = nn.CrossEntropyLoss()
            else:
                loss_fn = nn.MSELoss()

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

        # prediction heads for all layers, each with its own lr_layers[l]
        for l in range(self.total_layers):
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
        self.z_states = {}
        self.last_pred = {}
        self.h        = {}

        for l in range(self.total_layers):
            H = self.hidden_sizes[l]
            self.z_states[l] = torch.zeros(1, H, device=self.device)
            self.h[l]        = torch.zeros(1, 1, H, device=self.device)

            if l == 0:
                self.last_pred[l] = torch.zeros(1, 1, self.vocab_size,
                                                device=self.device)
            else:
                # initial prediction from zero z_ema
                self.last_pred[l] = self.layers[l].prediction(self.z_states[l], None)



    # ===================================================================
    def summary(self):
        print("\n===== Model Summary =====")
        print(f"Total layers: {self.total_layers}")
        print(f"Hidden sizes: {self.hidden_sizes}")
        print(f"Sleep interval: {self.sleep_interval}")
        print(f"Sleep steps: {self.sleep_steps}")
        print(f"Device: {self.device}")
        print("=================================\n")


    def wake_step(self, x, y):
        """
        Joint wake update:
        - L0 memory + prediction are trained every step.
        - Upper-layer prediction heads are trained when their horizon closes.
        - Context to each layer l is [pred_{l+1}, z_ema_{l+1}] (when l+1 exists).
        """

        self.step += 1
        t = self.step

        x = x.to(self.device)
        y = y.to(self.device)
        
        # ---------------- LAYER 0 memory FORWARD ----------------
        mem0_loss, _, _, _ = self.layers[0].train_memory(
                                                    x, threshold=self.threshold
                                                )


        # update L0 state (no grad restriction)
        with torch.no_grad():
            _, z0, _ = self.layers[0].memory(x, None)
        self.z_states[0]= z0
        #self.z_ema[0]   = self.ema_alpha * self.z_ema[0] + (1 - self.ema_alpha) * z0_seq

        # ------------- UPDATE UPPER STATES & NEW PREDICTIONS -------------
        for l in range(1, self.total_layers):
            stride = self.short_term_memory ** l
            if t % stride != 0:
                continue

            current_z = self.z_states[l - 1].unsqueeze(1)

            z, self.h[l] = self.layers[l].memory.encode_step_from_vec(
                current_z, self.h[l]
            )
            self.z_states[l] = z.squeeze(1)
        
        # ------------- UPDATE UPPER NEW PREDICTIONS -------------
        for l in reversed(range(1, self.total_layers)):
            current_z = self.z_states[l]

            # context from ABOVE: [pred_{l+1}, z_ema_{l+1}]
            if l + 1 < self.total_layers:
                ctx_pred = self.last_pred[l + 1]      # dim H_l
                ctx_mem  = self.z_states[l + 1]       # dim H_{l+1}
                ctx_l    = torch.cat([ctx_pred, ctx_mem], dim=-1)
            else:
                ctx_l = None

            pred_l = self.layers[l].prediction(current_z, ctx_l)
            self.last_pred[l] = pred_l

        if self.total_layers > 1:
            # pred_1 has dim H0, z_ema[1] has dim H1
            ctx_pred1 = self.last_pred[1]
            ctx_mem1  = self.z_states[1]
            #print(ctx_pred1.shape, ctx_mem1.shape)
            ctx0      = torch.cat([ctx_pred1, ctx_mem1], dim=-1)
        else:
            ctx0 = None

        pred0 = self.layers[0].prediction(z0, ctx0)
        pred_loss = self.layers[0].compute_pred_loss(pred0, y)

        # ------------- JOINT BACKWARD + STEP -------------
        self.optimizer_wake.zero_grad()
        pred_loss.backward()
        self.optimizer_wake.step()

        return dict(
            step=t,
            loss_mem=mem0_loss,
            loss_pred=pred_loss.item(),
            logits_pred0=pred0.detach(),
        )

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
            - EMA smoothing: stabilizes replay drift (slow waves, replay compression).
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
                input  = stacked EMA-smoothed window
                target = most recent EMA z
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
            - EMA stabilizes replay and avoids catastrophic drift.
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
        total_steps = self.sleep_steps[target_layer] * train_stride
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
                                        window, threshold=self.threshold
                                    )


        print('Sleeping memory loss ', loss)
