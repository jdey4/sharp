import torch
import torch.nn as nn
from torch import optim
from collections import deque
from .layer import Layer  
from ..utils.loss import CrossEntropyLayerLoss, MSELayerLoss



class Model(nn.Module):
    """
        Hierarchical predictive + memory model.
        Fully kwargs-driven: ANY hyperparameter can be passed at init.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # ============================================================
        # 1. DEFAULT SETTINGS
        # ============================================================
        defaults = dict(
            total_layers = 3,
            vocab_size = None,
            hidden_sizes = None,
            embedding_dim_l0 = None,
            short_term_memory = 3,
            tau = 0.5,

            # learning
            lr_layers = None,            # list of per-layer LRs
            optimizer_class = optim.Adam,
            optimizer_kwargs = None,     # dict: lr, betas, weight_decay, etc.

            # sleep dynamics
            ema_alpha = 0.3,
            sleep_interval = 1000,
            sleep_steps = None,          # dict {layer: steps}

            # misc
            device = "cpu",
        )

        # ============================================================
        # 2. MERGE DEFAULTS WITH USER KWARGS
        # ============================================================
        for k,v in {**defaults, **kwargs}.items():
            setattr(self, k, v)

        self.device = torch.device(self.device)

        # Default optimizer settings per layer
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {"lr": 1e-3, "weight_decay": 1e-8}

        # ============================================================
        # 3. VALIDITY CHECKS
        # ============================================================
        assert self.vocab_size is not None
        assert self.hidden_sizes is not None
        assert self.embedding_dim_l0 is not None
        assert self.lr_layers is not None
        assert len(self.hidden_sizes) == self.total_layers
        assert len(self.lr_layers) == self.total_layers

        if self.sleep_steps is None:
            self.sleep_steps = {l: 100 for l in range(1, self.total_layers)}

        # ============================================================
        # 4. BUILD LAYERS — WITH LOSS + OPTIMIZER PER LAYER
        # ============================================================
        self.layers = nn.ModuleList()

        for l in range(self.total_layers):

            # input + context sizes
            if l == 0:
                inp = self.vocab_size
                ctx = self.hidden_sizes[l+1] if self.total_layers > 1 else 0
                emb = self.embedding_dim_l0
            else:
                inp = self.hidden_sizes[l-1]
                ctx = self.hidden_sizes[l+1] if (l+1 < self.total_layers) else 0
                emb = None

            # --------------------------
            # Choose loss for this layer
            # --------------------------
            if l == 0:
                loss_fn = CrossEntropyLayerLoss()
            else:
                loss_fn = MSELayerLoss()

            # Use layer-specific LR
            local_opt_kwargs = dict(self.optimizer_kwargs)
            local_opt_kwargs["lr"] = self.lr_layers[l]

            # Build layer WITH internal optimizer
            self.layers.append(
                Layer(
                    input_size=inp,
                    hidden_size=self.hidden_sizes[l],
                    loss_function=loss_fn,
                    optimizer_class=self.optimizer_class,
                    optimizer_kwargs=local_opt_kwargs,
                    embedding_dim=emb,
                    layer=l,
                    context_size=ctx,
                    tau=self.tau
                ).to(self.device)
            )

        # ============================================================
        # 5. HIDDEN STATES, TARGETS, EMA BUFFERS
        # ============================================================
        self.z_states = {}
        self.z_targets = {}
        self.z_ema = {}
        self.last_pred = {}

        for l in range(self.total_layers):
            H = self.hidden_sizes[l]
            self.z_states[l] = torch.zeros(1, 1, H, device=self.device)
            self.z_ema[l] = torch.zeros(1, 1, H, device=self.device)

            if l == 0:
                self.z_targets[l] = torch.zeros(1, 1, device=self.device)
                self.last_pred[l] = torch.zeros(1, 1, self.vocab_size, device=self.device)
            else:
                self.z_targets[l] = torch.zeros(1, 1, self.hidden_sizes[l-1]).to(self.device)
                self.last_pred[l] = torch.zeros(1, 1, self.hidden_sizes[l-1]).to(self.device)



    # ===================================================================
    def summary(self):
        print("\n===== Model Summary =====")
        print(f"Total layers: {self.total_layers}")
        print(f"Hidden sizes: {self.hidden_sizes}")
        print(f"Sleep interval: {self.sleep_interval}")
        print(f"Sleep steps: {self.sleep_steps}")
        print(f"EMA alpha: {self.ema_alpha}")
        print(f"Device: {self.device}")
        print("=================================\n")


    def wake_step(self, x, y):
        r"""
            One wake-phase update.

            Behavior:
            - Layer 0:
                * trains (memory + prediction) on every step with gradient
                * updates z_states[0] and z_ema[0]

            - Layer l ≥ 1:
                * only updates its z_state (NO weight update) when
                        step % (short_term_memory ** l) == 0
                * input  = z_ema[l-1]      (downsampled snapshot from below)
                * context = z_states[l+1]  (if exists, else None)

            This implements a temporal hierarchy:
                L0: fast timescale
                L1: slower (every short_term_memory^1 steps)
                L2: even slower (every short_term_memory^2 steps)
        """
        if not hasattr(self, "wake_step_counter"):
            self.wake_step_counter = 0

        self.wake_step_counter += 1
        t = self.wake_step_counter

        x = x.to(self.device)
        y = y.to(self.device)

        # ============================================================
        # LAYER 0 — FAST LEARNING
        # ============================================================

        # Context for layer 0 is the prediction from layer 1 (if exists)
        if self.total_layers > 1:
            # last stored prediction of layer 1
            ctx0 = self.z_states[1]
        else:
            ctx0 = None

        # Train layer 0 normally
        loss0, rec0, pred0, z0, _ = self.layers[0].train_step(
            x, y, h0=None, context=ctx0
        )

        # update latent & EMA
        z0_seq = z0.unsqueeze(1)
        self.z_states[0] = z0_seq
        self.z_ema[0] = self.ema_alpha * self.z_ema[0] + (1 - self.ema_alpha) * z0_seq


        self.last_pred[0] = pred0    # shape (B,1,vocab)


        # ============================================================
        # UPPER LAYERS — NO LEARNING, ONLY STATE UPDATE
        # ============================================================

        with torch.no_grad():
            for l in range(1, self.total_layers):

                if l == 1:
                    stride = 1
                else:
                    stride = self.short_term_memory ** (l-1)

                if t % stride != 0:
                    continue

                # Memory input is EMA of below layer
                inp_l = self.z_ema[l-1]

                # Context is prediction from layer above
                if l + 1 < self.total_layers:
                    ctx_l = self.z_states[l+1]
                else:
                    ctx_l = None

                # — This call gives both z_l AND pred_l —
                with torch.no_grad():
                    _, pred_l, z_l, _ = self.layers[l](inp_l, h0=None, context=ctx_l)

                # update states
                z_l_seq = z_l.unsqueeze(1)
                self.z_states[l] = z_l_seq
                self.z_ema[l]    = self.ema_alpha * self.z_ema[l] + \
                                    (1 - self.ema_alpha) * z_l_seq

                # store for downstream context
                self.last_pred[l] = pred_l

        return dict(
            step=t,
            loss0=loss0,
            logits_rec0=rec0,
            logits_pred0=pred0
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


        source_layer = target_layer - 1


        # Initialize buffers
        H_lower = self.layers[source_layer].hidden_size
        stm_queue = deque(
            [torch.zeros(1, 1, H_lower, device=self.device) for _ in range(self.short_term_memory)],
            maxlen=self.short_term_memory
        )

        train_stride = self.short_term_memory

        # Initialize EMA hidden
        z_ema = torch.zeros(1, 1, H_lower, device=self.device)
        z_target = torch.zeros(1, 1, H_lower, device=self.device)

        # Training loop
        total_steps = self.sleep_steps[target_layer] * train_stride
        for t in range(total_steps):
            if t == 0:
                x_next, z, h = self.layers[source_layer].generate_sample()
            else:
                x_next, z, h = self.layers[source_layer].generate_sample(x=x_next, h0=h)

            # --- EMA smoothing before downsampling ---
            z_ema = self.ema_alpha * z_ema + (1 - self.ema_alpha) * z

            # --- Downsampling / training trigger ---
            if t % train_stride == 0:
                stm_queue.append(z_target.clone())
                z_target = z_ema.clone()
                window = torch.cat(list(stm_queue), dim=1)  # (1, stm, H_lower)
                loss, _, _, _, _ = self.layers[target_layer].train_step(
                                        window, z_target
                                    )


        print('Sleeping memory loss ', loss)
