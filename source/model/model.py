import torch
import torch.nn as nn
from torch import optim
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

        for l in range(self.total_layers):
            H = self.hidden_sizes[l]
            self.z_states[l] = torch.zeros(1, 1, H, device=self.device)
            self.z_ema[l] = torch.zeros(1, 1, H, device=self.device)

            if l == 0:
                self.z_targets[l] = torch.zeros(1, 1, device=self.device)
            else:
                self.z_targets[l] = torch.zeros(1, 1, self.hidden_sizes[l-1]).to(self.device)

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
