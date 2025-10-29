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

#%%
def main():
    # ---- Parameters (your style) ----
    total_samples, n_community, n_members = 1000000, 2, 3
    total_layers, short_term_memory = 2, 4

    vocab_size = n_community * n_members + 1
    hidden_size_memory = [60, 180, 540][:total_layers]
    emb_dim_l0 = 20

    # Explicit per-layer hidden sizes for prediction heads
    pred_hidden_sizes = hidden_size_memory #[60, 180, 540][:total_layers]  

    lr_memory = [1e-4] + [5e-5] * (total_layers - 1)
    lr_prediction = 1e-3
    alpha = 0.0
    sleep_interval_wake = 1000
    sleep_steps_per_L = {l: 1000 for l in range(1, total_layers)}

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
    pred_blocks, pred_criteria, pred_opts = {}, [], []
    for l in range(total_layers):
        ctx_size = hidden_size_memory[l] if (l + 1) < total_layers else 0
        out_size = vocab_size if l == 0 else hidden_size_memory[l - 1]
        pred_blocks[l] = Prediction(hidden_size_memory[l], pred_hidden_sizes[l], out_size, ctx_size)
        pred_criteria.append(
            nn.CrossEntropyLoss() if l==0 else nn.MSELoss()
        )
        pred_opts.append(torch.optim.Adam(pred_blocks[l].parameters(), lr=lr_prediction, weight_decay=1e-8))

    #print(mem_blocks, pred_blocks)
    # ---- Data ----
    data = get_sequence(total_samples, n_community, n_members, train_percent=1.0)
    dataset = DatasetConverter(data, working_memory=1, short_term_memory=short_term_memory)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- States ----
    h_states = {l: None for l in range(total_layers)}
    h_targets = {l: None for l in range(total_layers)}
    cntxt = {l: None for l in range(total_layers)}

    correct_ring = np.zeros(1000)
    total = 0

    for X, y in loader:
        total += 1
        # L0 AE always trains on the current short sequence X
        l0_ae_loss = train_memory_layer(mem_blocks[0], mem_opts[0], mem_criteria[0], X, layer=0)
        # Update L0 hidden from the current sequence
        
        # feature extraction only
        with torch.no_grad():
            _, h0 = mem_blocks[0](X)
            _, h0_next = mem_blocks[0](y, h0)
        h_states[0]  = h0                 # (1,1,H0)
        h_targets[1] = h0_next            # (1,1,H0)
            
        logits, loss = train_pattern_recognition(pred_blocks, 0, pred_opts[0], pred_criteria[0], h_states[0], 
                                            y, context=cntxt[1] if total_layers==2 else None)
            
        # Strided updates for upper layers: only update when total % layer_strides[l] == 0
        '''for l in range(1, total_layers):
            stride = short_term_memory ** l
            if (total % stride == 0):
                # Single-step encode from lower layer
                h_states[l] = mem_blocks[l].encode_step_from_vec(h_states[l-1], h_states[l])

                if l>1:
                    h_targets[l] = mem_blocks[l-1].encode_step_from_vec(h_targets[l-1], h_states[l-1])
                
                l_mse = train_pattern_recognition(pred_blocks, l, pred_opts[l], pred_criteria[l], h_states[l], 
                                            h_targets[l], context=cntxt[l+1] if l<total_layers-1 else None)
                cntxt[l] = pred_blocks[l](h_states[l], cntxt[l+1] if l<total_layers-1 else None)'''

        with torch.no_grad():
            pred_tok = logits.argmax(dim=-1)
            correct_ring[total % 1000] = (pred_tok[0, 0] == y[0, 0]).item()
            if total % 1000 == 0:
                acc = np.sum(correct_ring) / (1000 if total >= 1000 else total)
                print(f"Iter {total} | AE={l0_ae_loss:.4f} | CE={loss.item():.4f} | acc={acc:.4f}")
                #print("ctx norm:", cntxt[1].norm().item(),
                #        "| h1 norm:", h_states[1].norm().item(),
                #        "| mse l1:", l_mse.item())


        '''if total % sleep_interval_wake == 0:
            print("Entering sleep ...")
            for l in range(1, total_layers):
                print("Training Layer ", l)
                sleep_train_layer(l, sleep_steps_per_L[l], short_term_memory, mem_blocks, mem_opts, mem_criteria, pred_blocks, ema_alpha=alpha)'''



# %%
if __name__ == "__main__":
    main()

# %%
# ---- Parameters (your style) ----
total_samples, n_community, n_members = 500000, 2, 3
total_layers, short_term_memory = 3, 5

vocab_size = n_community * n_members + 1
hidden_size_memory = [60, 180, 540][:total_layers]
emb_dim_l0 = 20

# Explicit per-layer hidden sizes for prediction heads
pred_hidden_sizes = hidden_size_memory #[60, 180, 540][:total_layers]  

lr_memory = [1e-3] + [1e-3] * (total_layers - 1)
lr_prediction = 1e-3
alpha = 0.0
sleep_interval_wake = 10000
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
pred_blocks, pred_criteria, pred_opts = {}, [], []
for l in range(total_layers):
    ctx_size = hidden_size_memory[l] if (l + 1) < total_layers else 0
    out_size = vocab_size if l == 0 else hidden_size_memory[l - 1]
    pred_blocks[l] = Prediction(hidden_size_memory[l], pred_hidden_sizes[l], out_size, ctx_size)
    pred_criteria.append(
        nn.CrossEntropyLoss() if l==0 else nn.MSELoss()
    )
    pred_opts.append(torch.optim.Adam(pred_blocks[l].parameters(), lr=lr_prediction, weight_decay=1e-8))

#print(mem_blocks, pred_blocks)
# ---- Data ----
data = get_sequence(total_samples, n_community, n_members, train_percent=1.0)
dataset = DatasetConverter(data, working_memory=1, short_term_memory=short_term_memory)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ---- States ----
h_states = {l: None for l in range(total_layers)}
h_targets = {l: None for l in range(total_layers)}
cntxt = {l: None for l in range(total_layers)}

correct_ring = np.zeros(1000)
total = 0

for X, y in loader:
    total += 1
    # L0 AE always trains on the current short sequence X
    l0_ae_loss = train_memory_layer(mem_blocks[0], mem_opts[0], mem_criteria[0], X, layer=0)
    # Update L0 hidden from the current sequence
    
    # feature extraction only
    with torch.no_grad():
        _, h0 = mem_blocks[0](X)
        _, h0_next = mem_blocks[0](y, h0)
    h_states[0]  = h0                 # (1,1,H0)
    h_targets[1] = h0_next            # (1,1,H0)
        
    logits, loss = train_pattern_recognition(pred_blocks, 0, pred_opts[0], pred_criteria[0], h_states[0], 
                                        y, context=cntxt[0] if total_layers==2 else None)
        
    # Strided updates for upper layers: only update when total % layer_strides[l] == 0
    for l in range(1, total_layers):
        stride = short_term_memory ** l
        if (total % stride == 0):
            # Single-step encode from lower layer
            with torch.no_grad():
                h_states[l] = mem_blocks[l].encode_step_from_vec(h_states[l-1], h_states[l])

                if l>1:
                    h_targets[l] = mem_blocks[l-1].encode_step_from_vec(h_targets[l-1], h_states[l-1])
                
            l_mse = train_pattern_recognition(pred_blocks, l, pred_opts[l], pred_criteria[l], h_states[l], 
                                        h_targets[l], context=cntxt[l+1] if l<total_layers-1 else None)
            
            with torch.no_grad():
                cntxt[l-1] = pred_blocks[l](h_states[l], cntxt[l+1] if l<total_layers-1 else None)

    with torch.no_grad():
        pred_tok = logits.argmax(dim=-1)
        correct_ring[total % 1000] = (pred_tok[0, 0] == y[0, 0]).item()
        if total % 1000 == 0:
            acc = np.sum(correct_ring) / (1000 if total >= 1000 else total)
            print(f"Iter {total} | AE={l0_ae_loss:.4f} | CE={loss.item():.4f} | acc={acc:.4f}")
            print("ctx norm:", cntxt[0].norm().item(),
                    "| h1 norm:", h_states[1].norm().item(),
                    "| mse l1:", l_mse.item())


    if total % sleep_interval_wake == 0:
        print("Entering sleep ...")
        for l in range(1, total_layers):
            print("Training Layer ", l)
            sleep_train_layer(l, sleep_steps_per_L[l], short_term_memory, mem_blocks, mem_opts, mem_criteria, pred_blocks, sigma=0.5, ema_alpha=.5)

# %%
