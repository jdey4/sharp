# =========================
# Helpers
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def train_memory_layer(model, optimizer, criterion, X, layer=0):
    """
    Perform one supervised or self-reconstruction training step for a memory block.

    This function trains a given memory layer (`model`) to reconstruct or predict its
    input sequence `X`. It handles both the lowest-level (token-level) memory block
    and higher-level compressed layers differently, depending on the `layer` index.

    Args:
        model (nn.Module):
            The memory block to be trained (e.g., an RNN or autoencoder).
        optimizer (torch.optim.Optimizer):
            Optimizer associated with this memory block.
        criterion (nn.Module):
            Loss function used for reconstruction or prediction (e.g., MSELoss, CrossEntropyLoss).
        X (torch.Tensor):
            Input tensor or hidden state sequence used as both input and target.
            For the bottom layer, shape is typically `(batch, seq_len, vocab_dim or embed_dim)`.
        layer (int, optional):
            Index of the memory layer. If `0`, the model is trained token-by-token
            using averaged loss across time. If greater than 0, the model is trained
            to reconstruct the entire sequence at once.
            Default is `0`.

    Returns:
        torch.Tensor:
            Detached scalar loss tensor (no gradient attached), suitable for logging.

    Notes:
        - Sets the model to training mode.
        - Zeroes gradients, performs forward and backward pass, applies gradient clipping,
          and updates weights.
        - For `layer == 0`, computes per-timestep loss and averages across sequence length.
        - Gradient clipping prevents instability, especially for recurrent models.
    """
    with torch.enable_grad():
        model.train()
        optimizer.zero_grad()

        if layer == 0:
            logits, _ = model(X)
            loss = sum(criterion(logits[:, t], X[:, t]) for t in range(X.size(1))) / X.size(1)
        else:
            logits, _ = model(X)
            loss = criterion(logits, X)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.detach()

def freeze_range(model_blocks, start, end):
    """
    Freeze a contiguous range of memory or prediction layers, preventing weight 
    updates.

    Sets `requires_grad = False` for all parameters of the memory and prediction 
    blocks between indices `start` and `end` (inclusive). This is typically used 
    during the sleep/replay phase to ensure lower layers remain fixed while higher 
    layers are being trained.

    Args:
        model_blocks (list[nn.Module]):
            List or container of model blocks (e.g., RNNs, autoencoders, etc.).
        start (int):
            Starting layer index (inclusive) to freeze.
        end (int):
            Ending layer index (inclusive) to freeze.

    Example:
        >>> freeze_range(mem_blocks, 0, 2)
        # Freezes layers 0, 1, and 2.
    """
    for l in range(start, end + 1):
        for p in model_blocks[l].parameters():
            p.requires_grad = False




def unfreeze_range(model_blocks, start, end):
    """
    Unfreeze (re-enable gradient updates) for a contiguous range of memory and prediction
    layers.

    Sets `requires_grad = True` for all parameters of memory and prediction blocks 
    between indices `start` and `end` (inclusive). This restores trainability after 
    a period of frozen operation (e.g., after the sleep-phase replay training).

    Args:
        model_blocks (list[nn.Module]):
            List or container of memory or prediction block modules.
        start (int):
            Starting layer index (inclusive) to unfreeze.
        end (int):
            Ending layer index (inclusive) to unfreeze.

    Example:
        >>> unfreeze_range(mem_blocks, 0, 2)
        # Unfreezes layers 0, 1, and 2.
    """
    for l in range(start, end + 1):
        for p in model_blocks[l].parameters():
            p.requires_grad = True
    


# =========================
# Train Pattern Recognition Blocks 
# =========================
def train_pattern_recognition(
        pred_blocks, optimizer, criteria,
        h_input, h_target, context
):
    """
    Jointly trains all hierarchical prediction (generative) heads with shared gradients.

    This function performs a *coupled* optimization step across all predictive layers
    of the hierarchy.  Each prediction head `pred_blocks[l]` receives its current-layer
    hidden state `h_input[l]` and an optional top-down `context[l]`, and produces a
    prediction that is compared against a supervision target.  The total multi-layer
    loss is accumulated and back-propagated jointly through all layers, allowing
    cross-layer gradient flow.

    Parameters
    ----------
    pred_blocks : list[nn.Module]
        List or dict-like container of `Prediction` modules, one per hierarchical layer.
        Each block maps its current hidden representation (and optional context) to a
        target output—typically a token distribution for layer 0 and hidden-state
        predictions for higher layers.

    optimizer : torch.optim.Optimizer
        Optimizer that manages parameters of *all* prediction heads simultaneously.
        Coupled optimization ensures global gradient flow and joint temporal credit
        assignment across layers.

    criteria : list[nn.Module]
        Loss functions per layer.  For example:
        - `nn.CrossEntropyLoss()` for layer 0 (token prediction)
        - `nn.MSELoss()` for higher layers (hidden-state prediction)

    h_input : list[torch.Tensor]
        List of current hidden states, one per layer.  
        `h_input[l]` has shape `(B, T, H_l)` or `(B, 1, H_l)` depending on layer setup.

    h_target : torch.Tensor or list[torch.Tensor]
        Target signal(s) for supervision.  For the bottom layer this is usually the next
        token index, while upper layers may receive next-step hidden states or other
        continuous targets.  A single tensor may be broadcast to all layers if shared.

    context : list[torch.Tensor or None]
        Optional contextual inputs from higher layers.  
        If provided, `context[l]` is concatenated with `h_input[l]` inside each
        prediction head.  May be `None` for topmost layer.

    Returns
    -------
    logits : torch.Tensor
        Output logits of the final layer processed in the loop (often layer 0).
        Returned detached from the graph for logging or evaluation.

    loss : torch.Tensor
        Detached scalar total loss value (sum of per-layer losses).

    Notes
    -----
    • **Coupled optimization**:
      Unlike per-layer training, this routine accumulates all layer losses before
      performing a single backward pass.  This allows gradients to propagate through
      the entire predictive hierarchy, aligning generation weights at all levels.

    • **Analogy to wake-sleep learning**:
      - Memory / encoder modules act as *recognition weights* (bottom-up inference).
      - Prediction heads act as *generative weights* (top-down reconstruction).
      Joint training makes the system behave like a continuous wake-sleep phase,
      where recognition and generation paths co-adapt in the same gradient step.

    • **Typical layer semantics**:
      | Layer | Input                     | Target                     | Loss |
      |-------|---------------------------|-----------------------------|------|
      | 0     | token-level hidden state  | next token label            | CE   |
      | 1     | compressed hidden state   | next lower-layer hidden     | MSE  |
      | ...   | ...                       | ...                         | ...  |

    Example
    -------
    >>> logits, loss = train_pattern_recognition(
    ...     pred_blocks, optimizer, criteria,
    ...     h_input=[h0, h1], h_target=y, context=[ctx0, ctx1]
    ... )
    >>> print(f"Total predictive loss: {loss.item():.4f}")
    """

    total_layers = len(pred_blocks)
    loss = 0.0
    with torch.enable_grad():
        optimizer.zero_grad()
        for l in range(total_layers):
            pred_blocks[l].train()
            if l == 0:
                logits = pred_blocks[0](h_input[l], context[l])
                layer_loss = criteria[0](logits[0, 0], h_target[0, 0])
            else:
                logits_ = pred_blocks[l](h_input[l], context[l])
                layer_loss = criteria[l](logits_, h_target)
            loss += layer_loss

        loss.backward()
        optimizer.step()

        return logits.detach(), loss.detach()


# =========================
# Sleep replay (for a layer-pair)
# =========================
def sleep_train_layer(
    target_layer, replay_steps, short_term_memory,
    mem_blocks, mem_opts, mem_criteria, 
    pred_blocks, sigma=0.00, ema_alpha=0.1
):
    """
    Performs sleep-phase replay and hierarchical consolidation for the specified target layer.

    During the sleep phase, the lower (source) layer generates synthetic hidden-state
    trajectories (or token-driven sequences) which act as replayed experience.
    The target (upper) layer learns to compress or reconstruct these replay patterns,
    enabling slow consolidation of fast-learned episodic traces into long-term memory.

    This function emulates the hippocampal-cortical consolidation process seen in
    wake-sleep algorithms and biological replay systems. It stabilizes generated
    dynamics via exponential moving average (EMA) smoothing and optionally injects
    stochastic noise to mimic spontaneous reactivation variability (e.g., SWRs).

    Args:
        target_layer (int): 
            Index of the memory layer to train during sleep.
            The layer below (target_layer - 1) is treated as the replay source.

        replay_steps (int): 
            Number of synthetic replay steps to generate from the source layer.

        short_term_memory (int): 
            Temporal window length (number of recent hidden states) to use as
            input for training the target layer's memory block.

        mem_blocks (list[nn.Module]): 
            List of hierarchical memory (RNN/autoencoder) modules. 
            Only the target layer is trained; lower layers are frozen during replay.

        mem_opts (list[torch.optim.Optimizer]): 
            Optimizers corresponding to each memory block.

        mem_criteria (list[Callable]): 
            Loss functions for each memory block, typically MSE for reconstruction.

        pred_blocks (list[nn.Module]): 
            List of associated prediction heads for each layer used to generate
            next-step states or tokens during replay.

        sigma (float, optional): 
            Standard deviation of Gaussian replay noise (default: 0.0).
            Adds variability to generated hidden states, encouraging robustness.

        ema_alpha (float, optional): 
            Exponential moving average coefficient (default: 0.1).
            Higher values produce stronger smoothing and slower adaptation.

    Returns:
        None
            Trains the target layer's memory block in-place. Prints the final replay loss.

    Process Overview:
        1. Freeze source (lower) layer modules to prevent gradient flow.
        2. Initialize replay generator (hidden state) and short-term memory buffer.
        3. Iteratively generate synthetic hidden sequences:
            - For layer 0: sample tokens from softmax distribution.
            - For higher layers: propagate hidden states via predictor.
        4. Apply EMA smoothing to stabilize replay signals.
        5. Every `short_term_memory` steps, downsample the replayed sequence,
           concatenate recent hidden windows, and train the target layer using MSE loss.
        6. Unfreeze source layer modules after completion.

    Biological Analogy:
        - Source layer ≈ hippocampus (fast learner, replay generator)
        - Target layer ≈ cortex (slow learner, consolidator)
        - Replay trajectories ≈ hippocampal reactivation during sleep
        - EMA smoothing ≈ replay drift stabilization
        - Noise (sigma) ≈ stochastic variability in neural replay
    """

    source_layer = target_layer - 1
    device = next(mem_blocks[source_layer].parameters()).device

    # Freeze lower layers during replay
    freeze_range(mem_blocks, source_layer, source_layer)
    freeze_range(pred_blocks, source_layer, source_layer)

    # Initialize buffers
    H_lower = mem_blocks[source_layer].hidden_size
    stm_queue = deque(
        [torch.zeros(1, 1, H_lower, device=device) for _ in range(short_term_memory)],
        maxlen=short_term_memory
    )

    upper_mb, upper_opt, upper_crit = (
        mem_blocks[target_layer],
        mem_opts[target_layer],
        mem_criteria[target_layer],
    )
    train_stride = short_term_memory

    # Initialize hidden generators
    h_gen = torch.zeros(1, 1, mem_blocks[source_layer].hidden_size, device=device)

    # Initialize EMA hidden
    h_ema = torch.zeros(1, 1, mem_blocks[source_layer].hidden_size, device=device)
    h_target = torch.zeros(1, 1, mem_blocks[source_layer].hidden_size, device=device)

    # Training loop
    total_steps = replay_steps * train_stride
    for t in range(1, total_steps + 1):
        with torch.no_grad():
            if source_layer == 0:
                logits0 = pred_blocks[0](h_gen)
                probs0 = torch.softmax(logits0[0, 0], dim=-1)
                token = torch.multinomial(probs0, num_samples=1)
                h_gen = mem_blocks[0].encode_step_from_token(token, h_gen)
            else:
                #up_ctx = pred_blocks[target_layer](h_gen[target_layer])
                pred_lower = pred_blocks[source_layer](h_gen)
                if sigma > 0:
                    pred_lower = pred_lower + sigma * torch.randn_like(pred_lower)
                h_gen = mem_blocks[source_layer].encode_step_from_vec(
                                                    pred_lower, h_gen
                                                )

        # --- EMA smoothing before downsampling ---
        h_ema = ema_alpha * h_ema + (1 - ema_alpha) * h_gen

        # --- Downsampling / training trigger ---
        if t % train_stride == 0:
            stm_queue.append(h_target.clone())
            h_target = h_ema.clone()
            window = torch.cat(list(stm_queue), dim=1)  # (1, stm, H_lower)
            mem_loss = train_memory_layer(upper_mb, upper_opt, upper_crit, window, layer=target_layer)

    
    print('Sleeping memory loss ', mem_loss)
    # Unfreeze lower layers
    unfreeze_range(mem_blocks, source_layer, source_layer)

