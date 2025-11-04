#%%
from source.utils import get_sequence, DatasetConverter
from source.model.memory import Memory
from source.model.prediction import Prediction
from source.model.helpers import train_memory_layer,\
    sleep_train_layer, train_pattern_recognition

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import itertools
from collections import deque
# %%
# ---- Parameters (your style) ----
total_samples, n_community, n_members = 500000, 2, 3
total_layers, short_term_memory = 3, 3

vocab_size = n_community * n_members + 1
hidden_size_memory = [60, 180, 540, 1000][:total_layers]
emb_dim_l0 = 30

# Explicit per-layer hidden sizes for prediction heads
pred_hidden_sizes = hidden_size_memory #[60, 180, 540][:total_layers]  

lr_memory = [1e-3] + [1e-3] * (total_layers - 1)
grad_eps = [1e-3, 3e-2, 1e-1]
pred_eps = 5e-2
lr_prediction = 4e-4
ema_alpha = 0.5
sleep_interval_wake = 30000
sleep_steps_per_L = {1:10000, 2:1000} #{l: 1000 for l in range(1, total_layers)}

# ---- per-layer wake-time strides ----
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
pred_blocks, pred_criteria = {}, []
for l in range(total_layers):
    ctx_size = hidden_size_memory[l] if (l + 1) < total_layers else 0
    out_size = vocab_size if l == 0 else hidden_size_memory[l - 1]
    pred_blocks[l] = Prediction(hidden_size_memory[l], pred_hidden_sizes[l], out_size, ctx_size)
    pred_criteria.append(
        nn.CrossEntropyLoss() if l==0 else nn.MSELoss()
    )
    #pred_opts.append(torch.optim.Adam(pred_blocks[l].parameters(), lr=lr_prediction, weight_decay=1e-8))
pred_opt = torch.optim.Adam(itertools.chain(*[p.parameters() for p in pred_blocks.values()]),
                                lr=lr_prediction, weight_decay=1e-8)

#print(mem_blocks, pred_blocks)
# ---- Data ----
data = get_sequence(total_samples, n_community, n_members, train_percent=1.0)
dataset = DatasetConverter(data, working_memory=1, short_term_memory=short_term_memory)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ---- States ----
h_states = {l: None for l in range(total_layers)}
h_targets = {l: None for l in range(total_layers)}
#cntxt = {l: None for l in range(total_layers-1)}
h_ema = {l: None for l in range(total_layers)}

for ii in range(total_layers):
    h_ema[ii] = torch.zeros(1, 1, hidden_size_memory[ii])
#%%
correct_ring = np.zeros(1000)
total = 0

for X, y in loader:
    # L0 AE always trains on the current short sequence X
    l0_ae_loss = train_memory_layer(mem_blocks[0], mem_opts[0], mem_criteria[0], X, layer=0, eps=grad_eps[0])
    # Update L0 hidden from the current sequence
    
    # feature extraction only
    with torch.no_grad():
        _, h0 = mem_blocks[0](X)
        h_states[0]  = h0                 # (1,1,H0)
        h_targets[0] = y

        if total_layers>1 and total % layer_strides[1] == 0:
            _, h0_next = mem_blocks[0](torch.cat((X[:,1:],y), dim=1))
            h_targets[1] = h0_next            # (1,1,H0)

        # Strided updates for upper layers: only update when total % layer_strides[l] == 0
        h_ema[0] = ema_alpha * h_ema[0] + (1 - ema_alpha) * h_states[0]

        for l in range(1, total_layers):
            stride = layer_strides[l]
            if (total % stride == 0):
                # Single-step encode from upper layer
                with torch.no_grad():
                    #print(l, h_states[l])
                    h_states[l] = mem_blocks[l].encode_step_from_vec(h_ema[l-1], h_states[l])

                    if l+1 < total_layers:
                        h_targets[l+1] = mem_blocks[l].encode_step_from_vec(h_targets[l], h_states[l])
                        
                    h_ema[l] = ema_alpha * h_ema[l] + (1 - ema_alpha) * h_states[l]

    logits, loss = train_pattern_recognition(
                                pred_blocks, pred_opt, pred_criteria, 
                                h_states, h_targets, alpha=1e-3, eps=pred_eps
                            )
        
    

    with torch.no_grad():
        total += 1
        pred_tok = logits.argmax(dim=-1)
        correct_ring[total % 1000] = (pred_tok[0, 0] == y[0, 0]).item()
        if total % 1000 == 0:
            acc = np.sum(correct_ring) / (1000 if total >= 1000 else total)
            print(f"Iter {total} | AE={l0_ae_loss:.4f} | CE={loss.item():.4f} | acc={acc:.4f}")
            
    if total % sleep_interval_wake == 0:
            print("Entering sleep ...")
            for l in range(1, total_layers):
                print("Training Layer ", l)
                sleep_train_layer(l, sleep_steps_per_L[l], short_term_memory, mem_blocks, mem_opts, mem_criteria, pred_blocks, sigma=0.0, ema_alpha=ema_alpha, eps=grad_eps[l])

# %%
