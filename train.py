#%%
from source.utils import get_sequence, DatasetConverter
from source.model.model import Model

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
sleep_interval_wake = 30000
total_samples, n_community, n_members = 1000000, 2, 3
total_layers, short_term_memory = 1, 7

vocab_size = n_community * n_members + 1

data = get_sequence(total_samples, n_community, n_members, train_percent=1.0)#/(n_members/10.0))


dataset = DatasetConverter(data, working_memory=1, short_term_memory=short_term_memory)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# ============================================================
# Build a 3-layer hierarchical predictive + memory model
# ============================================================
model = Model(
    total_layers = total_layers,

    # ---- Layer sizes ----
    vocab_size = vocab_size,                  # layer 0 input dimension
    hidden_sizes = [60],    # H0, H1, H2
    embedding_dim_l0 = 30,

    # ---- Learning rates per layer ----
    lr_layers = [1e-3],   

    # ---- Optimizer type (user can choose) ----
    optimizer_class = torch.optim.Adam,
    optimizer_kwargs = {
        "weight_decay": 1e-8
    },

    # ---- Sleep hyperparameters ----
    short_term_memory = 3,
    ema_alpha = 0.1,
    sleep_interval = 1000,
    sleep_steps = {1: 1000, 2: 1000},   # layer 2 is the top

    # ---- Misc ----
    tau = 0.8,
    device = device
)

model.summary()

ii = 0 
correct_ring = np.zeros(1000)
for x, y in loader:
    #loss, _, _, _, _ = model.layers[0].train_step(x,y)
    loss = model.wake_step(x,y)


    with torch.no_grad():
        ii += 1
        pred_tok = loss['logits_pred0'].argmax(dim=-1)
        correct_ring[ii % 1000] = (pred_tok[0] == y[0, 0]).item()
        
        if ii%1000 == 0:
            acc = np.sum(correct_ring) / (1000 if ii >= 1000 else ii)
            print("Iter ", ii, "prediction loss: ", loss['loss_pred'], "memory loss: ", loss['loss_mem'], "Acc: ", acc)

    '''if ii % sleep_interval_wake == 0:
        print("Entering sleep ...")
        for _ in range(1):
            for layer in range(1, model.total_layers):
                model.sleep_train_layer(
                        target_layer=layer
                    )'''
                #print("Layer ",layer, " sleep loss: ", sleep_loss)

# %%
