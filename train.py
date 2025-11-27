#%%
from source.utils import get_sequence, DatasetConverter
from source.utils import CrossEntropyL1Loss, MSEL1Loss
from source.model.model import Model
from source.model.helpers import sleep_train_layer

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
    device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Using device:", device)

    # ---- Parameters (your style) ----
    total_samples, n_community, n_members = 100000, 2, 3
    total_layers, short_term_memory = 3, 3

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
        hidden_sizes = [60, 180, 540],    # H0, H1, H2
        embedding_dim_l0 = 30,

        # ---- Learning rates per layer ----
        lr_layers = [1e-3, 1e-3, 1e-3],   

        # ---- Optimizer type (user can choose) ----
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = {
            "weight_decay": 1e-8
        },

        # ---- Sleep hyperparameters ----
        short_term_memory = 3,
        ema_alpha = 0.3,
        sleep_interval = 1000,
        sleep_steps = {1: 100, 2: 100},   # layer 2 is the top

        # ---- Misc ----
        tau = 0.6,
        device = device
    )

    model.summary()

    ii = 0 
    for x, y in loader:
        loss, _, _, _, _ = model.layers[0].train_step(x,y)

        ii += 1

        if ii%1000 == 0:
            print("Iter ", ii, " loss: ", loss)
    



# %%
if __name__ == "__main__":
    main()

#%%
device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Using device:", device)

# ---- Parameters (your style) ----
total_samples, n_community, n_members = 100000, 2, 3
total_layers, short_term_memory = 3, 3

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
    hidden_sizes = [60, 180, 540],    # H0, H1, H2
    embedding_dim_l0 = 30,

    # ---- Learning rates per layer ----
    lr_layers = [1e-3, 1e-3, 1e-3],   

    # ---- Optimizer type (user can choose) ----
    optimizer_class = torch.optim.Adam,
    optimizer_kwargs = {
        "weight_decay": 1e-8
    },

    # ---- Sleep hyperparameters ----
    short_term_memory = 3,
    ema_alpha = 0.3,
    sleep_interval = 1000,
    sleep_steps = {1: 100, 2: 100},   # layer 2 is the top

    # ---- Misc ----
    tau = 0.9,
    device = device
)

model.summary()

ii = 0 
for x, y in loader:
    loss, _, _, _, _ = model.layers[0].train_step(x,y)

    ii += 1

    if ii%1000 == 0:
        print("Iter ", ii, " loss: ", loss)

# %%
