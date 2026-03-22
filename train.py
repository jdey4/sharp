#%%
from sharp.utils import get_sequence, DatasetConverter
from sharp.model.model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import itertools 
from collections import deque

#%%
device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Using device:", device)

# ---- Parameters ----
total_samples, n_community, n_members, context_depth = 5000000, 2, 3, 26
total_layers, short_term_memory = 3, 4

vocab_size = 27

data = get_sequence(total_samples, n_community, n_members, context_depth=context_depth, train_percent=0.33, direction_mode="hash_parity")


dataset = DatasetConverter(data, short_term_memory=short_term_memory)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# ============================================================
# Build a 3-layer hierarchical predictive + memory model
# ============================================================
model = Model(
    total_layers = total_layers,

    # ---- Layer sizes ----
    vocab_size = vocab_size,                  # layer 0 input dimension
    hidden_sizes = [64, 128, 256],    # H0, H1, H2
    embedding_dim_l0 = 30,

    # ---- Learning rates per layer ----
    lr_layers = 1e-4,   

    # ---- Optimizer type (user can choose) ----
    optimizer_class = torch.optim.Adam,
    optimizer_kwargs = {
        "weight_decay": 1e-12
    },

    # ---- Sleep hyperparameters ----
    short_term_memory = short_term_memory,
    context_tag_buffer_size=100,
    # ---- Misc ----
    device = device
)

model.summary()

#%%
model.reset_model()

ii = 0 
h_ = None
correct_ring = np.zeros(1000)
for x, y in loader:
    #loss, _, _, _, _ = model.layers[0].train_step(x,y)
    logits, loss, recon_loss, h_ = model.wake_step(x, y, h_)


    with torch.no_grad():
        ii += 1
        pred_tok = logits.argmax(dim=-1)
        correct_ring[ii % 1000] = (pred_tok[0] == y[0, 0]).item()
        
        if ii%1000 == 0:
            acc = np.sum(correct_ring) / (1000 if ii >= 1000 else ii)
            print("Iter ", ii, f"prediction loss: {loss:.8e}", f"Memory loss: {recon_loss:.8e}", "Acc: ", acc)
            if model.sleeping:
                print("Sleep on ", model.recon_loss_ema)

    if ii%20000==0:
        model.sleep(total_steps=1000)
 # %%

for jj in range(model.context_tag_buffer_size):
    seq = ''
    h = model.context_tags[jj][0].unsqueeze(0)
    for ii in range(64):
        h, x = model._teacher_step_layer0(h, context=model.context_tags[jj][1])
        
        seq += chr(int(x.item()) + ord('A'))

    print(seq)

# %%
