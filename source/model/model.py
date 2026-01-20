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
            tau = 0.1,
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
        
        assert self.vocab_size is not None
        assert self.hidden_sizes is not None
        assert self.embedding_dim_l0 is not None
        assert self.lr_layers is not None
        assert len(self.hidden_sizes) == self.total_layers
        assert len(self.lr_layers) == self.total_layers

        if self.sleep_steps is None:
            self.sleep_steps = {l: 100 for l in range(1, self.total_layers)}

        self.step = 1
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
                emb = self.embedding_dim_l0

            # context from layer l+1:
            #   - prediction dimension  = hidden_sizes[l]
            #   - memory z dim      = hidden_sizes[l+1]
            if l + 1 < self.total_layers:
                ctx = self.hidden_sizes[l]
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
                    optimizer_class = self.optimizer_class,   
                    optimizer_kwargs= self.optimizer_kwargs, 
                    embedding_dim= emb,
                    layer       = l,
                    context_size= ctx,
                    tau         = self.tau,
                ).to(self.device)
            )

        # ------------------------------------------------------------
        # 5. STATE 
        # ------------------------------------------------------------
        self.h_states = {}
        self.last_pred = {}
        self.h_pass = {}

        for l in range(self.total_layers):
            H = self.hidden_sizes[l]
            self.h_states[l] = torch.zeros(1, H, device=self.device)
            self.h_pass[l] = None

            if l == 0:
                self.last_pred[l] = torch.zeros(1, self.vocab_size,
                                                device=self.device)
            else:
                # initial prediction from zero h_states
                with torch.no_grad():
                    self.last_pred[l] = self.layers[l].prediction(self.h_states[l], None)



    # ===================================================================
    def summary(self):
        print("\n===== Model Summary =====")
        print(f"Total layers: {self.total_layers}")
        print(f"Hidden sizes: {self.hidden_sizes}")
        print(f"Sleep steps: {self.sleep_steps}")
        print(f"Device: {self.device}")
        print("=================================\n")

    
    def wake_step(self, x, y):
        """
        """

        self.step += 1
        t = self.step

        x = x.to(self.device)
        y = y.to(self.device)

        
        for l in range(self.total_layers):
            stride = self.short_term_memory ** l
            if t % stride != 0:
                continue
            
            if l!=self.total_layers-1:
                cntx = self.last_pred[l+1]
            else:
                cntx = None

            if l!=0:
                ### Handle Prediction Blocks ###
                _, _ = self.layers[l].train_prediction(
                                        self.h_states[l], self.h_states[l-1], 
                                        cntx,
                                        threshold=0
                                    )
                
            ### Handle Memory Blocks ###
            if l==0:
                loss_recon, logits_rec0, h, self.h_pass[0] = self.layers[0].train_memory(
                                        x, self.h_pass[0],
                                        threshold=self.threshold
                                    )
            else:
                h = self.layers[l].memory.encode_step_from_vec(
                                        self.h_states[l-1], self.h_states[l]
                                    ).detach()

            self.h_states[l] = h

            ### Handle Prediction Blocks ###
            if l == 0:
                loss_pred, self.last_pred[0] = self.layers[0].train_prediction(
                                        self.h_states[0], y, 
                                        cntx, threshold=0
                                    )
            else:
                with torch.no_grad():
                    self.last_pred[l] = self.layers[l].prediction(
                                            self.h_states[l], 
                                            cntx
                                        )

        
        return dict(
            step=t,
            loss_mem=loss_recon,
            loss_pred=loss_pred,
            logits_rec0=logits_rec0.detach(),
            logits_pred0=self.last_pred[0].detach(),
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
        h_ = None
        target_h = None
        total_steps = self.sleep_steps * train_stride
        for t in range(total_steps):
            if t == 0:
                x_next, h = self.layers[source_layer].generate_sample()
            else:
                x_next, h = self.layers[source_layer].generate_sample(x=x_next, h0=h)

            
            # --- Downsampling / training trigger ---
            if t % train_stride == 0:
                if target_h != None:
                    stm_queue.append(target_h.clone())
                
                target_h = h.clone()

                window = torch.cat(list(stm_queue), dim=1)  # (1, stm, H_lower)
                loss_recon, _, h_, self.h_pass[target_layer] = self.layers[target_layer].train_memory(
                                        window, self.h_pass[target_layer], threshold=self.threshold
                                    )
                
                '''loss_pred, _ = self.layers[target_layer].train_prediction(
                                        h_.squeeze(1), target_h.squeeze(1), 
                                        threshold=self.threshold
                                    )'''

            if t%5000 == 0:
                print('Sleeping memory loss ', loss_recon, ' at layer ', target_layer)
