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

def freeze_range(mem_blocks, start, end):
    """
    Freeze a contiguous range of memory layers, preventing weight updates.

    Sets `requires_grad = False` for all parameters of the memory blocks between
    indices `start` and `end` (inclusive). This is typically used during the
    sleep/replay phase to ensure lower layers remain fixed while higher layers
    are being trained.

    Args:
        mem_blocks (list[nn.Module]):
            List or container of memory blocks (e.g., RNNs, autoencoders, etc.).
        start (int):
            Starting layer index (inclusive) to freeze.
        end (int):
            Ending layer index (inclusive) to freeze.

    Example:
        >>> freeze_range(mem_blocks, 0, 2)
        # Freezes layers 0, 1, and 2.
    """
    for l in range(start, end + 1):
        for p in mem_blocks[l].parameters():
            p.requires_grad = False


def unfreeze_range(mem_blocks, start, end):
    """
    Unfreeze (re-enable gradient updates) for a contiguous range of memory layers.

    Sets `requires_grad = True` for all parameters of memory blocks between indices
    `start` and `end` (inclusive). This restores trainability after a period of
    frozen operation (e.g., after the sleep-phase replay training).

    Args:
        mem_blocks (list[nn.Module]):
            List or container of memory block modules.
        start (int):
            Starting layer index (inclusive) to unfreeze.
        end (int):
            Ending layer index (inclusive) to unfreeze.

    Example:
        >>> unfreeze_range(mem_blocks, 0, 2)
        # Unfreezes layers 0, 1, and 2.
    """
    for l in range(start, end + 1):
        for p in mem_blocks[l].parameters():
            p.requires_grad = True


def build_contexts_topdown(pred_blocks, h_dict, total_layers):
    """
    Construct hierarchical top-down context signals for each layer.

    Given a set of prediction modules (`pred_blocks`) and their corresponding hidden
    states (`h_dict`), this function computes top-down contextual signals layer by
    layer — starting from the topmost layer down to the lower layers.

    The resulting dictionary `ctx` contains the predictive context at each layer,
    which can be used during both wake (forward) and sleep (replay) phases.

    Args:
        pred_blocks (list[nn.Module]):
            List of predictive modules that map hidden states to context vectors.
            Typically, each block takes `(hidden, upper_context)` as input.
        h_dict (dict[int, torch.Tensor or None]):
            Dictionary of current hidden states for each layer.
            Layers with `None` will be skipped.
        total_layers (int):
            Total number of layers in the hierarchy.

    Returns:
        dict[int, torch.Tensor or None]:
            Dictionary mapping layer index → computed context tensor.
            The top layer produces context from its own hidden state; lower layers
            receive context modulated by upper-layer predictions.

    Notes:
        - Starts from the top layer (`total_layers - 1`) and propagates context downward.
        - If a layer has no hidden state (`h_dict[l] is None`), its context is `None`.
        - If an upper context exists (`ctx[l + 1]`), it is passed into the prediction block.
        - This implements hierarchical top-down modulation similar to cortical feedback.

    Example:
        >>> ctx = build_contexts_topdown(pred_blocks, h_dict, total_layers=3)
        >>> print(ctx[1].shape)  # Context vector for layer 1
    """
    ctx = {}
    top = total_layers - 1
    ctx[top] = pred_blocks[top](h_dict[top]) if h_dict[top] is not None else None

    for l in range(top - 1, 0, -1):
        if h_dict[l] is None:
            ctx[l] = None
        else:
            if ctx[l + 1] is not None:
                ctx[l] = pred_blocks[l](h_dict[l], ctx[l + 1])
            else:
                ctx[l] = pred_blocks[l](h_dict[l])
    return ctx

# =========================
# Train Pattern Recognition Blocks 
# =========================

def train_pattern_recognition(
    pred_blocks, pred_layer, optimizer, criterion,
    h, target, context=None
):
    """
    Perform one supervised training step for a prediction block within the hierarchical model.

    This function trains a single prediction module (`pred_blocks[pred_layer]`) to map
    its current hidden state (and optionally contextual input) to a target output.
    It executes a standard forward-backward optimization pass and returns the detached
    loss tensor for logging.

    Args:
        pred_blocks (list[nn.Module]):
            A list (or dict-like container) of prediction modules.
            Each element maps hidden representations to output predictions.

        pred_layer (int):
            Index of the prediction block to be trained.

        optimizer (torch.optim.Optimizer):
            Optimizer corresponding to `pred_blocks[pred_layer]`.
            It is assumed that it manages only that layer’s parameters.

        criterion (nn.Module):
            Loss function used for supervision (e.g., CrossEntropyLoss or MSELoss).

        h (torch.Tensor):
            Input hidden state tensor for this prediction layer.
            Shape typically `(batch, time, hidden_dim)` or `(batch, hidden_dim)`.

        target (torch.Tensor):
            Ground-truth labels or regression targets compatible with `criterion`.

        context (torch.Tensor, optional):
            Optional contextual input from a higher or lower layer.
            Default is `None`.

    Returns:
        torch.Tensor:
            The detached scalar loss tensor (no gradient attached),
            suitable for logging or monitoring.

    Example:
        >>> loss = train_pattern_recognition(pred_blocks, 0, optimizer, loss_fn, h, y)
        >>> print(f"Training loss: {loss.item():.4f}")

    Notes:
        - The function sets the target prediction module into training mode (`.train()`),
          ensuring layers like dropout or batch normalization behave appropriately.
        - Gradients are zeroed, forward pass is computed, loss is backpropagated, and
          parameters are updated in place.
        - Returning `loss.detach()` allows the caller to log the loss without maintaining
          computational graph references.
    """

    with torch.enable_grad():
        pred_blocks[pred_layer].train()
        optimizer.zero_grad()

        #print(h.shape, context.shape if context is not None else 'None', 'context')
        logits = pred_blocks[pred_layer](h, context)
        #print(logits[0,0], target[0,0])

        if pred_layer == 0:
            loss = criterion(logits[0, 0], target[0, 0])
        else:
            loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

    if pred_layer == 0:
        return logits, loss.detach()
    else:
        return loss.detach()


# =========================
# Sleep replay (for a layer-pair)
# =========================
def sleep_train_layer(
    target_layer, replay_steps, short_term_memory,
    mem_blocks, mem_opts, mem_criteria, pred_blocks,
    sigma=0.00, ema_alpha=0.1
):
    """
    Sleep-phase replay training for hierarchical memory layers with EMA smoothing.

    Args:
        target_layer: int, which layer to train.
        replay_steps: number of replay cycles per stride.
        short_term_memory: memory window length.
        mem_blocks, mem_opts, mem_criteria, pred_blocks: module lists.
        total_layers: total number of layers.
        sigma: noise level for replay (default 0.0).
        ema_alpha: exponential moving average factor (default 0.1).
    """
    source_layer = target_layer - 1
    device = next(mem_blocks[source_layer].parameters()).device

    # Freeze lower layers during replay
    freeze_range(mem_blocks, 0, source_layer)

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
    h_gen = {
        l: torch.zeros(1, 1, mem_blocks[l].hidden_size, device=device)
        for l in [source_layer, target_layer]
    }

    # Initialize EMA hidden
    h_ema = torch.zeros_like(h_gen[source_layer])

    total_steps = replay_steps * train_stride
    for t in range(1, total_steps + 1):
        with torch.no_grad():
            if source_layer == 0:
                ctx1 = pred_blocks[1](h_gen[1])
                logits0 = pred_blocks[0](h_gen[0], ctx1)
                probs0 = torch.softmax(logits0[0, 0], dim=-1)
                token = torch.multinomial(probs0, num_samples=1)
                h_gen[0] = mem_blocks[0].encode_step_from_token(token, h_gen[0])
            else:
                up_ctx = pred_blocks[target_layer](h_gen[target_layer])
                pred_lower = pred_blocks[source_layer](h_gen[source_layer], up_ctx)
                if sigma > 0:
                    pred_lower = pred_lower + sigma * torch.randn_like(pred_lower)
                h_gen[source_layer] = mem_blocks[source_layer].encode_step_from_vec(
                    pred_lower, h_gen[source_layer]
                )

        # --- EMA smoothing before downsampling ---
        h_ema = ema_alpha * h_ema + (1 - ema_alpha) * h_gen[source_layer]

        # --- Downsampling / training trigger ---
        if t % train_stride == 0:
            stm_queue.append(h_ema.clone())
            window = torch.cat(list(stm_queue), dim=1)  # (1, stm, H_lower)
            loss = train_memory_layer(upper_mb, upper_opt, upper_crit, window, layer=target_layer)

            #if t % 100 == 0:
            #    print('Sleeing loss ', loss)
    
    print('Sleeping loss ', loss)
    # Unfreeze lower layers
    unfreeze_range(mem_blocks, 0, source_layer)

