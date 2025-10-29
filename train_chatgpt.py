#%%
import os, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

# --- local imports (use your library) ---
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)

from source.utils import get_sequence, DatasetConverter
from source.model.memory import Memory
from source.model.prediction import Prediction
from source.model.helpers import train_memory_layer, sleep_train_layer, train_pattern_recognition

#%%
# --------------------- config ---------------------
total_samples, n_community, n_members = 500_000, 2, 3
total_layers, short_term_memory = 3, 5

vocab_size = n_community * n_members + 1
hidden_size_memory = [60, 180, 540][:total_layers]
emb_dim_l0 = 20
pred_hidden_sizes = hidden_size_memory

lr_memory = [1e-3] + [1e-3] * (total_layers - 1)
lr_prediction = 1e-3
weight_decay = 1e-8

# sleep schedule
sleep_interval_wake = 10_000
sleep_steps_per_L = {1: 10_000, 2: 1_000}

# wake strides (L1 every 5, L2 every 25)
base_stride = short_term_memory
layer_strides = [1] + [base_stride ** l for l in range(1, total_layers)]
print(f"[config] layer_strides (wake): {layer_strides}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- build modules ---------------------
mem_blocks, mem_criteria, mem_opts = {}, [], []
for l in range(total_layers):
    if l == 0:
        mb = Memory(vocab_size, hidden_size_memory[l], embedding_dim=emb_dim_l0, layer=0).to(device)
        mem_criteria.append(nn.CrossEntropyLoss())
    else:
        mb = Memory(hidden_size_memory[l-1], hidden_size_memory[l], layer=l).to(device)
        mem_criteria.append(nn.MSELoss())
    mem_blocks[l] = mb
    mem_opts.append(torch.optim.Adam(mb.parameters(), lr=lr_memory[l], weight_decay=weight_decay))

pred_blocks, pred_criteria, pred_opts = {}, [], []
for l in range(total_layers):
    ctx_size = hidden_size_memory[l] if (l + 1) < total_layers else 0
    out_size = vocab_size if l == 0 else hidden_size_memory[l - 1]
    pb = Prediction(hidden_size_memory[l], pred_hidden_sizes[l], out_size, ctx_size).to(device)
    pred_blocks[l] = pb
    pred_criteria.append(nn.CrossEntropyLoss() if l == 0 else nn.MSELoss())
    pred_opts.append(torch.optim.Adam(pb.parameters(), lr=lr_prediction, weight_decay=weight_decay))

# --------------------- data ---------------------
data = get_sequence(total_samples, n_community, n_members, train_percent=1.0)
dataset = DatasetConverter(data, working_memory=1, short_term_memory=short_term_memory)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# --------------------- states ---------------------
# h_states[l]      = current hidden h_l(t)
# h_nextlow[l]     = target for head-l: next lower hidden h_{l-1}(t+1)  (for l>=1)
# ctx_for[l]       = top-down context to feed into head-l (produced by head-(l+1))
h_states  = {l: None for l in range(total_layers)}
h_nextlow = {l: None for l in range(total_layers)}
ctx_for   = {l: None for l in range(total_layers)}

correct_ring = np.zeros(1000, dtype=np.float32)
total = 0

# --------------------- training loop ---------------------
for X, y in loader:
    total += 1
    X = X.to(device)   # (1, stm)
    y = y.to(device)   # (1, 1)

    # 0) Wake AE for L0 (tokens)
    l0_ae = train_memory_layer(mem_blocks[0], mem_opts[0], mem_criteria[0], X, layer=0)

    # 1) Extract L0 features (no grad): h0(t) and h0(t+1)
    with torch.no_grad():
        _, h0_t = mem_blocks[0](X)         # (1,1,H0)
        _, h0_n = mem_blocks[0](y, h0_t)   # next lower for head-1
    h_states[0]  = h0_t
    h_nextlow[1] = h0_n

    # 2) Update upper hidden states on their strides and build next-lower chain correctly
    #    h_nextlow[1] = h0(t+1)
    #    h_nextlow[2] = Mem1( input = h_nextlow[1], h_prev = h1(t) ) -> h1(t+1)
    for l in range(1, total_layers):
        stride = layer_strides[l]
        if (total % stride) == 0:
            with torch.no_grad():
                # current hidden at layer l using current lower h_{l-1}(t)
                h_states[l] = mem_blocks[l].encode_step_from_vec(h_states[l-1], h_states[l])
                if l >= 2:
                    # build next-lower for head-l by advancing layer (l-1) with input = h_nextlow[l-1]
                    h_nextlow[l] = mem_blocks[l-1].encode_step_from_vec(h_nextlow[l-1], h_states[l-1])

    # 3) Build top-down contexts for ALL layers (no grad), top→down
    with torch.no_grad():
        ctx_for = {l: None for l in range(total_layers)}
        for l in range(total_layers - 2, -1, -1):  # l = L-2,...,0
            upper = l + 1
            if h_states[upper] is not None:
                up_ctx = ctx_for[upper]  # None at top-1
                ctx_for[l] = pred_blocks[upper](h_states[upper], up_ctx)

    # 4) Train upper heads (use their own context), only when their stride fired
    for l in range(total_layers - 1, 0, -1):
        stride = layer_strides[l]
        if (total % stride) == 0 and (h_states[l] is not None) and (h_nextlow[l] is not None):
            feat_l = h_states[l].detach()
            tgt_l  = h_nextlow[l].detach()
            ctx_l  = ctx_for[l]  # context FOR layer l
            _ = train_pattern_recognition(
                pred_blocks, l, pred_opts[l], pred_criteria[l],
                feat_l, tgt_l, context=ctx_l
            )

    # 5) Train head-0 (token CE) using fresh context from layer-1
    feat0 = h_states[0].detach()
    ctx0  = ctx_for[0]  # produced this step
    logits, ce0 = train_pattern_recognition(
        pred_blocks, 0, pred_opts[0], pred_criteria[0],
        feat0, y, context=ctx0
    )

    # 6) Metrics
    with torch.no_grad():
        pred_tok = logits.argmax(dim=-1)
        correct_ring[total % 1000] = (pred_tok[0, 0] == y[0, 0]).item()
        if total % 1000 == 0:
            acc = correct_ring.mean() if total >= 1000 else correct_ring[: total % 1000].mean()
            extras = []
            if ctx_for[0] is not None: extras.append(f"ctx0‖={ctx_for[0].norm().item():.3f}")
            if h_states[1] is not None: extras.append(f"h1‖={h_states[1].norm().item():.3f}")
            print(f"Iter {total} | L0-AE={l0_ae:.4f} | L0-CE={ce0.item():.4f} | acc={acc:.4f}"
                  + (" | " + " ".join(extras) if extras else ""))

    # 7) Sleep (uses safe helper below; stride = STM)
    if total % sleep_interval_wake == 0:
        print("Entering sleep ...")
        for l in range(1, total_layers):
            steps = sleep_steps_per_L.get(l, 0)
            if steps > 0:
                print("  Sleep-train Layer", l)
                sleep_train_layer(
                    target_layer=l,
                    replay_steps=steps,
                    short_term_memory=short_term_memory,
                    mem_blocks=mem_blocks,
                    mem_opts=mem_opts,
                    mem_criteria=mem_criteria,
                    pred_blocks=pred_blocks,
                    sigma=0.05,      # small diversity
                    ema_alpha=0.7    # faster EMA (new = 0.7*new + 0.3*old)
                )

print("Training finished.")

# %%
