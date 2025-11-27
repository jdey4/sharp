# =========================
# Helpers
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# =========================
# Sleep replay (for a layer-pair)
# =========================
def sleep_train_layer(
    model, target_layer, replay_steps, short_term_memory
):
    """
    Sleep-phase replay for hierarchical consolidation in the model.

    This function trains ONLY the specified `target_layer` using replayed
    latent trajectories generated from the *lower* layer (target_layer - 1),
    analogous to hippocampus→cortex consolidation in the brain.

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
    model : Model
        Hierarchical model containing all layers, weights, and optimizers.

    target_layer : int
        Index of the layer to train during sleep.
        The source layer is (target_layer - 1).

    replay_steps : int
        Number of synthetic replay steps to generate.
        Actual internal steps = replay_steps * short_term_memory.

    short_term_memory : int
        Size of temporal context window for the target layer.
        Acts as the sleep-time “bPTT window”.

    ─────────────────────────────────────────────────────────────
    RETURNS
    ─────────────────────────────────────────────────────────────
    None
        Prints final sleep replay loss for the target layer.

    ─────────────────────────────────────────────────────────────
    KEY PROPERTIES OF THIS IMPLEMENTATION
    ─────────────────────────────────────────────────────────────
    • Uses z (sampled variational latent) as replay representation.
    • Replay is context-free → matches hippocampal spontaneous replay.
    • Lower layers never receive gradients (natural freezing).
    • Upper layers learn stable long-timescale patterns.
    • Windowing produces temporal receptive fields (place-field-like tuning).
    • EMA stabilizes replay and avoids catastrophic drift.
    """


    source_layer = target_layer - 1
    device = model.device


    # Initialize buffers
    H_lower = model.layers[source_layer].hidden_size
    stm_queue = deque(
        [torch.zeros(1, 1, H_lower, device=device) for _ in range(short_term_memory)],
        maxlen=short_term_memory
    )

    train_stride = short_term_memory

    # Initialize EMA hidden
    z_ema = torch.zeros(1, 1, H_lower, device=device)
    z_target = torch.zeros(1, 1, H_lower, device=device)

    # Training loop
    total_steps = replay_steps * train_stride
    for t in range(1, total_steps + 1):
        if t == 0:
            x_next, z, h = model.layers[source_layer].generate_sample()
        else:
            x_next, z, h = model.layers[source_layer].generate_sample(x=x_next, h0=h)

        # --- EMA smoothing before downsampling ---
        z_ema = model.ema_alpha * z_ema + (1 - model.ema_alpha) * z

        # --- Downsampling / training trigger ---
        if t % train_stride == 0:
            stm_queue.append(z_target.clone())
            z_target = z_ema.clone()
            window = torch.cat(list(stm_queue), dim=1)  # (1, stm, H_lower)
            loss, _, _, _, _ = model.layers[target_layer].train_step(
                                    window, z_target
                                )


    print('Sleeping memory loss ', loss)
    
