#%%
import sys
sys.path.append('../..')
from source.utils import get_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import itertools
from collections import deque
#%%

# =========================
# Dataset
# =========================

class DatasetConverter(Dataset):
    def __init__(self, data, working_memory=1, short_term_memory=3):
        self.X = np.zeros((len(data)-working_memory-short_term_memory, short_term_memory), dtype=np.int64)
        self.y = np.zeros((len(data)-working_memory-short_term_memory, 1), dtype=np.int64)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.X[i, j] = ord(data[i+j]) - 65
            self.y[i] = ord(data[i+j+1]) - 65
        self.X = tnsr(self.X).long()
        self.y = tnsr(self.y).long()

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]


# =========================
# Modules
# =========================

class Memory(nn.Module):
    """Autoencoder-style memory block."""
    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size

        if layer == 0:
            assert embedding_dim is not None, "embedding_dim required for layer 0"
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True)
            dec_in = input_size
        else:
            self.encoder = nn.RNN(input_size, hidden_size, batch_first=True)
            dec_in = input_size

        self.decoder = nn.RNN(dec_in, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, dec_in)

        # init
        for name, p in self.encoder.named_parameters():
            if "weight_hh" in name: nn.init.orthogonal_(p)
            elif "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "bias" in name: nn.init.zeros_(p)
        for name, p in self.decoder.named_parameters():
            if "weight_hh" in name: nn.init.orthogonal_(p)
            elif "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, x, h0=None):
        if self.layer == 0:
            if x.dtype in (torch.int64, torch.int32):
                x_emb = self.embedding(x)
            else:
                x_emb = x
            _, h = self.encoder(x_emb, h0)
        else:
            _, h = self.encoder(x, h0)

        B, T = x.shape[0], x.shape[1]
        dec_in = torch.zeros((B, 1, self.input_size), device=x.device, dtype=torch.float)
        outs, h_dec = [], h
        for _ in range(T):
            d, h_dec = self.decoder(dec_in, h_dec)
            logits = self.out(d)
            outs.append(logits)
            dec_in = logits.detach()
        return torch.cat(outs, dim=1), h

    def encode_step_from_token(self, token_id, h_prev):
        assert self.layer == 0, "encode_step_from_token only for layer 0"
        emb = self.embedding(token_id.view(1, 1))
        _, h_next = self.encoder(emb, h_prev)
        return h_next

    def encode_step_from_vec(self, x_vec, h_prev):
        # x_vec must be (1,1,input_size_of_this_layer)
        _, h_next = self.encoder(x_vec, h_prev)
        return h_next


class Prediction(nn.Module):
    """Top-down prediction head."""
    def __init__(self, input_size, hidden_size, output_size, context_size=0):
        super().__init__()
        self.context_size = context_size
        self.l1 = nn.Linear(input_size + context_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, h, context=None):
        if self.context_size > 0:
            if context is None:
                context = torch.zeros(h.size(0), h.size(1), self.context_size,
                                      device=h.device, dtype=h.dtype)
            x_in = torch.cat((h, context), dim=2)
        else:
            x_in = h
        x = F.relu(self.l1(x_in))
        return self.l2(x)


# =========================
# Helpers
# =========================

def train_memory_layer(model, optimizer, criterion, X, layer=0):
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
    for l in range(start, end + 1):
        for p in mem_blocks[l].parameters():
            p.requires_grad = False

def unfreeze_range(mem_blocks, start, end):
    for l in range(start, end + 1):
        for p in mem_blocks[l].parameters():
            p.requires_grad = True

def build_contexts_topdown(pred_blocks, h_dict, total_layers):
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
# Sleep replay (layer-wise)
# =========================

def sleep_train_from_source(
    source_level, replay_steps, short_term_memory,
    mem_blocks, mem_opts, mem_criteria, pred_blocks,
    h_states, total_layers, sigma=0.05
):
    target_upper = source_level + 1
    if target_upper >= total_layers:
        return

    freeze_range(mem_blocks, 0, total_layers - 1)
    unfreeze_range(mem_blocks, target_upper, target_upper)

    h_gen = {l: (h_states[l].clone() if h_states[l] is not None
                 else torch.zeros(1, 1, mem_blocks[l].hidden_size))
             for l in range(total_layers)}

    H_lower = mem_blocks[target_upper - 1].hidden_size
    stm_queue = deque([torch.zeros(1, 1, H_lower) for _ in range(short_term_memory)],
                      maxlen=short_term_memory)

    upper_mb, upper_opt, upper_crit = mem_blocks[target_upper], mem_opts[target_upper], mem_criteria[target_upper]
    train_stride = max(1, short_term_memory)

    tokens = []
    for t in range(replay_steps):
        with torch.no_grad():
            if source_level == 0:
                ctx1 = pred_blocks[1](h_gen[1]) if total_layers > 1 else None
                logits0 = pred_blocks[0](h_gen[0], None)# ctx1)
                probs0 = torch.softmax(logits0[0, 0], dim=-1)
                token = torch.multinomial(probs0, num_samples=1)
                tokens.append(chr(token.item() + 65))
                h_gen[0] = mem_blocks[0].encode_step_from_token(token, h_gen[0])
            else:
                up_ctx = pred_blocks[source_level + 1](h_gen[source_level + 1]) \
                         if (source_level + 1) < total_layers else None
                pred_lower = pred_blocks[source_level](h_gen[source_level], up_ctx)
                if sigma > 0:
                    pred_lower = pred_lower + sigma * torch.randn_like(pred_lower)
                # pred_lower is (1,1,H_{source_level-1})
                h_gen[source_level] = mem_blocks[source_level].encode_step_from_vec(pred_lower, h_gen[source_level])

            # propagate upward one step
            '''for l in range(source_level + 1, total_layers):
                # feed the *current* lower hidden as a single-step vector
                h_gen[l] = mem_blocks[l].encode_step_from_vec(h_gen[l - 1], h_gen[l])'''

        if (t % train_stride) == 0:
            stm_queue.append(h_gen[target_upper - 1].detach().clone())
            window = torch.cat(list(stm_queue), dim=1)  # (1, stm, H_{target_upper-1})
            _ = train_memory_layer(upper_mb, upper_opt, upper_crit, window, layer=target_upper)

    unfreeze_range(mem_blocks, 0, total_layers - 1)
    if tokens:
        print(tokens)

# =========================
# Main
# =========================

def main():
    # ---- Parameters (your style) ----
    total_samples, n_community, n_members = 1000000, 2, 5
    total_layers, short_term_memory = 3, 5

    vocab_size = n_community * n_members + 1
    hidden_size_memory = [60, 180, 540, 1000][:total_layers]
    emb_dim_l0 = 20

    # Explicit per-layer hidden sizes for prediction heads
    pred_hidden_sizes = [60, 180, 540, 1000][:total_layers]  # freely change if you like

    lr_memory = [1e-4] + [5e-5] * (total_layers - 1)
    lr_prediction = 1e-3
    sleep_interval_wake = 10000
    sleep_steps_per_L = {l: 1000 for l in range(1, total_layers)}

    # ---- NEW: per-layer wake-time strides ----
    # layer_strides[L] applies to updating h_states[L] from h_states[L-1] during WAKE.
    # L0 is driven every step by tokens, so set stride 1 there.
    base_stride = short_term_memory  # you can pick any base; this is a reasonable default
    layer_strides = [1] + [base_stride ** l for l in range(1, total_layers)]
    # Example for total_layers=3, short_term_memory=4 -> [1, 4, 16]
    print(f"[config] layer_strides (wake): {layer_strides}")

    # ---- Memory blocks ----
    mem_blocks, mem_criteria, mem_opts = {}, [], []
    for l in range(total_layers):
        if l == 0:
            mem_blocks[l] = Memory(vocab_size, hidden_size_memory[l], embedding_dim=emb_dim_l0, layer=0)
            mem_criteria.append(nn.CrossEntropyLoss())
        else:
            mem_blocks[l] = Memory(hidden_size_memory[l - 1], hidden_size_memory[l], layer=l)
            mem_criteria.append(nn.MSELoss())
        mem_opts.append(torch.optim.Adam(mem_blocks[l].parameters(), lr=lr_memory[l], weight_decay=1e-8))

    # ---- Prediction heads ----
    pred_blocks = {}
    for l in range(total_layers):
        ctx_size = hidden_size_memory[l] if (l + 1) < total_layers else 0
        out_size = vocab_size if l == 0 else hidden_size_memory[l - 1]
        pred_blocks[l] = Prediction(hidden_size_memory[l], pred_hidden_sizes[l], out_size, ctx_size)

    pred_opt = torch.optim.Adam(itertools.chain(*[p.parameters() for p in pred_blocks.values()]),
                                lr=lr_prediction, weight_decay=1e-8)

    # ---- Data ----
    data = get_sequence(total_samples, n_community, n_members, train_percent=1.0/(n_members))
    dataset = DatasetConverter(data, working_memory=1, short_term_memory=short_term_memory)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- States ----
    h_states = {l: None for l in range(total_layers)}
    correct_ring = np.zeros(1000)
    total = 0

    # ---- Training (WAKE with strided upper-layer updates) ----
    for X, y in loader:
        total += 1

        # L0 AE always trains on the current short sequence X
        l0_ae_loss = train_memory_layer(mem_blocks[0], mem_opts[0], mem_criteria[0], X, layer=0)

        # Update L0 hidden from the current sequence
        with torch.no_grad():
            _, h_states[0] = mem_blocks[0](X)  # h_states[0] shape (1,1,H0)

            # Strided updates for upper layers: only update when total % layer_strides[l] == 0
            for l in range(1, total_layers):
                if (total % layer_strides[l] == 0) and (h_states[l-1] is not None):
                    # Single-step encode from lower hidden (already shape (1,1,H_{l-1}))
                    h_states[l] = mem_blocks[l].encode_step_from_vec(h_states[l-1], h_states[l])
                # else: keep previous h_states[l] as is (or None until first update)

        # Build contexts (top-down). Some h_states may still be None early on; handled inside.
        ctx = build_contexts_topdown(pred_blocks, h_states, total_layers)

        # L0 CE loss (requires h_states[0])
        logits0 = pred_blocks[0](h_states[0], ctx.get(1, None))
        ce_loss = F.cross_entropy(logits0[0], y[0])

        # Auxiliary alignment loss (only where both sides exist)
        aux = 0.0
        for l in range(1, total_layers):
            if (h_states[l] is not None) and (h_states[l-1] is not None):
                up_ctx = ctx.get(l + 1, None)
                pred = pred_blocks[l](h_states[l], up_ctx)
                aux += F.mse_loss(pred, h_states[l-1])

        loss = ce_loss + 1e-3 * aux

        pred_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(itertools.chain(*[p.parameters() for p in pred_blocks.values()]), 1.0)
        pred_opt.step()

        with torch.no_grad():
            pred_tok = logits0.argmax(dim=-1)
            correct_ring[total % 1000] = (pred_tok[0, 0] == y[0, 0]).item()
            if total % 1000 == 0:
                acc = np.sum(correct_ring) / (1000 if total >= 1000 else total)
                print(f"Iter {total} | AE={l0_ae_loss:.4f} | CE={ce_loss.item():.4f} | AUX={float(aux):.4f} | acc={acc:.4f}")

        # ---- Sleep (unchanged) ----
        if total % sleep_interval_wake == 0:
            print("Entering sleep ...")
            with torch.no_grad():
                for l in range(1, total_layers):
                    if h_states[l] is None and h_states[l - 1] is not None:
                        h_states[l] = mem_blocks[l].encode_step_from_vec(h_states[l - 1], None)

            for src in range(total_layers - 1):
                steps = sleep_steps_per_L.get(src + 1, 0)
                if steps > 0:
                    print(f"  Sleep: source L{src} -> train L{src+1} ({steps} steps)")
                    sleep_train_from_source(src, steps, short_term_memory,
                                            mem_blocks, mem_opts, mem_criteria,
                                            pred_blocks, h_states, total_layers,
                                            sigma=0.0)

    print("Training complete.")


if __name__ == "__main__":
    main()

# %%
