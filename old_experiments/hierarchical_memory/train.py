import sys
sys.path.append('..')
from source.utils import get_sequence, compute_bpc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import itertools
from collections import deque

# =========================
# Data
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
    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size

        if layer == 0:
            assert embedding_dim is not None
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
        dec_dtype = x.dtype if self.layer > 0 else torch.float
        dec_in = torch.zeros((B,1,self.input_size), device=x.device, dtype=dec_dtype)
        outs, h_dec = [], h
        for _ in range(T):
            d, h_dec = self.decoder(dec_in, h_dec)
            logits = self.out(d)
            outs.append(logits)
            dec_in = logits.detach()
        return torch.cat(outs, dim=1), h

    def encode_step_from_token(self, token_id, h_prev):
        emb = self.embedding(token_id.view(1,1))
        _, h_next = self.encoder(emb, h_prev)
        return h_next

    def encode_step_from_vec(self, x_vec, h_prev):
        _, h_next = self.encoder(x_vec, h_prev)
        return h_next


class Prediction(nn.Module):
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
# Training helpers
# =========================

def train_memory_layer(model, optimizer, criterion, X, layer=0):
    model.train()
    optimizer.zero_grad()
    if layer == 0:
        logits, _ = model(X)
        loss = sum(criterion(logits[:,t], X[:,t]) for t in range(X.size(1))) / X.size(1)
    else:
        logits, _ = model(X)
        loss = criterion(logits, X)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach()

@torch.no_grad()
def freeze_range(mem_blocks, start, end):
    for l in range(start, end+1):
        for p in mem_blocks[l].parameters():
            p.requires_grad = False

@torch.no_grad()
def unfreeze_range(mem_blocks, start, end):
    for l in range(start, end+1):
        for p in mem_blocks[l].parameters():
            p.requires_grad = True


# NOTE: removed @torch.no_grad() here to allow gradient flow
def build_contexts_topdown(pred_blocks, h_dict, total_layers):
    ctx = {}
    top = total_layers - 1
    ctx[top] = pred_blocks[top](h_dict[top]) if h_dict[top] is not None else None
    for l in range(top-1, 0, -1):
        if h_dict[l] is None:
            ctx[l] = None
        else:
            ctx[l] = pred_blocks[l](h_dict[l], ctx[l+1]) if ctx[l+1] is not None else pred_blocks[l](h_dict[l])
    return ctx


# =========================
# Sleep replay
# =========================

def sleep_train_upper_layer_with_context(
    target_upper, replay_steps, short_term_memory,
    mem_blocks, mem_opts, mem_criteria, pred_blocks,
    h_states, total_layers, sigma=0.02
):
    assert target_upper >= 1
    freeze_range(mem_blocks, 0, total_layers-1)
    unfreeze_range(mem_blocks, target_upper, target_upper)

    h_gen = {l: h_states[l] if h_states[l] is not None
             else torch.zeros(1,1,mem_blocks[l].hidden_size)
             for l in range(total_layers)}

    H_lower = mem_blocks[target_upper-1].hidden_size
    stm_queue = deque([torch.zeros(1,1,H_lower) for _ in range(short_term_memory)],
                      maxlen=short_term_memory)

    train_stride = max(1, (short_term_memory) ** target_upper)
    upper_mb, upper_opt, upper_crit = mem_blocks[target_upper], mem_opts[target_upper], mem_criteria[target_upper]

    for t in range(replay_steps):
        ctx = build_contexts_topdown(pred_blocks, h_gen, total_layers)

        l0_ctx = ctx.get(1, None)
        if l0_ctx is not None and sigma > 0:
            l0_ctx = l0_ctx + sigma * torch.randn_like(l0_ctx)

        logits0 = pred_blocks[0](h_gen[0], l0_ctx)
        probs0 = torch.softmax(logits0[0,0], dim=-1)
        x_t = torch.multinomial(probs0, num_samples=1)
        h_gen[0] = mem_blocks[0].encode_step_from_token(x_t, h_gen[0])

        for l in range(1, total_layers):
            stride_l = max(1, (short_term_memory-1) ** l)
            if (t % stride_l) == 0:
                inp = h_gen[l-1].transpose(0,1)
                h_gen[l] = mem_blocks[l].encode_step_from_vec(inp, h_gen[l])

        if (t % train_stride) == 0:
            stm_queue.append(h_gen[target_upper-1].detach().clone())
            window = torch.cat(list(stm_queue), dim=1)
            loss = train_memory_layer(upper_mb, upper_opt, upper_crit, window, layer=target_upper)
            #if (t % max(1, train_stride*5)) == 0:
            #    print(f"    [sleep t={t}] Train L{target_upper} | MSE={loss.item():.4f}")

    unfreeze_range(mem_blocks, 0, total_layers-1)


# =========================
# Main
# =========================

def main():
    total_samples, n_community, n_members = 1000000, 2, 4
    total_layers, short_term_memory = 3, 4

    vocab_size = n_community * n_members + 1
    hidden_size_memory = [100, 300, 1000]
    emb_dim_l0, pred_hidden = 30, 100
    lr_memory = [1e-4, 5e-5, 5e-5]
    lr_prediction = 1e-3
    sleep_interval_wake = 2000
    sleep_steps_per_L = {1: 1000, 2: 1000}

    # ---- Memory blocks ----
    mem_blocks, mem_criteria, mem_opts = {}, [], []
    for l in range(total_layers):
        if l == 0:
            mem_blocks[l] = Memory(vocab_size, hidden_size_memory[l], embedding_dim=emb_dim_l0, layer=0)
            mem_criteria.append(nn.CrossEntropyLoss())
        else:
            mem_blocks[l] = Memory(hidden_size_memory[l-1], hidden_size_memory[l], layer=l)
            mem_criteria.append(nn.MSELoss())
        mem_opts.append(torch.optim.Adam(mem_blocks[l].parameters(), lr=lr_memory[l], weight_decay=1e-8))

    # ---- Prediction heads ----
    pred_blocks = {}
    for l in range(total_layers):
        if l == 0:
            ctx_size = hidden_size_memory[0] if total_layers > 1 else 0
            out_size = vocab_size
        else:
            ctx_size = hidden_size_memory[l] if (l+1) < total_layers else 0
            out_size = hidden_size_memory[l-1]
        pred_blocks[l] = Prediction(hidden_size_memory[l], pred_hidden, out_size, ctx_size)
    pred_opt = torch.optim.Adam(itertools.chain(*[p.parameters() for p in pred_blocks.values()]),
                                lr=lr_prediction, weight_decay=1e-8)

    # ---- Data ----
    data = get_sequence(total_samples, n_community, n_members, train_percent=1.0)
    dataset = DatasetConverter(data, working_memory=1, short_term_memory=short_term_memory)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- States ----
    h_states = {l: None for l in range(total_layers)}
    correct_ring = np.zeros(1000)
    total = 0

    # ---- Training ----
    for X, y in loader:
        total += 1

        # Wake: train L0 memory AE
        l0_ae_loss = train_memory_layer(mem_blocks[0], mem_opts[0], mem_criteria[0], X, layer=0)

        # Update hidden hierarchy
        with torch.no_grad():
            _, h_states[0] = mem_blocks[0](X)
            for l in range(1, total_layers):
                if h_states[l-1] is not None:
                    inp = h_states[l-1].transpose(0,1)
                    h_states[l] = mem_blocks[l].encode_step_from_vec(inp, h_states[l])

        # Build contexts WITH grad
        ctx = build_contexts_topdown(pred_blocks, h_states, total_layers)
        l0_ctx = ctx.get(1, None)
        if l0_ctx is not None:
            assert l0_ctx.requires_grad, "ctx[1] should require grad for backprop"

        # L0 CE + auxiliary MSE for upper heads
        logits0 = pred_blocks[0](h_states[0], ctx.get(1, None))
        ce_loss = F.cross_entropy(logits0[0], y[0])
        aux = 0.0
        for l in range(1, total_layers):
            target = h_states[l-1]
            up_ctx = ctx.get(l+1, None)
            pred = pred_blocks[l](h_states[l], up_ctx)
            aux += F.mse_loss(pred, target)
        loss = ce_loss + 0.0 * aux

        pred_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(itertools.chain(*[p.parameters() for p in pred_blocks.values()]), 1.0)
        pred_opt.step()

        with torch.no_grad():
            pred_tok = logits0.argmax(dim=-1)
            correct_ring[total % 1000] = (pred_tok[0,0] == y[0,0]).item()
            if total % 1000 == 0:
                acc = np.sum(correct_ring) / (1000 if total >= 1000 else total)
                print(f"Iter {total} | AE={l0_ae_loss:.4f} | CE={ce_loss.item():.4f} | acc={acc:.4f}")

        # ---- Sleep ----
        if total % sleep_interval_wake == 0:
            print("Entering sleep ...")
            for l in range(1, total_layers):
                if h_states[l] is None and h_states[l-1] is not None:
                    inp = h_states[l-1].transpose(0,1)
                    h_states[l] = mem_blocks[l].encode_step_from_vec(inp, None)
            for upper in range(1, total_layers):
                steps = sleep_steps_per_L.get(upper, 0)
                if steps > 0:
                    print(f"  Sleep train L{upper} ({steps} steps)")
                    sleep_train_upper_layer_with_context(
                        upper, steps, short_term_memory,
                        mem_blocks, mem_opts, mem_criteria,
                        pred_blocks, h_states, total_layers, sigma=0.0
                    )

    print("Training complete.")


if __name__ == "__main__":
    main()
