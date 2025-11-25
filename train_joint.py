#%%
from source.utils import get_sequence, DatasetConverter
from source.utils import CrossEntropyL1Loss, MSEL1Loss
from source.model.model import Model
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
    device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Using device:", device)

    # ---- Parameters (your style) ----
    total_samples, n_community, n_members = 10000000, 2, 3
    total_layers, short_term_memory = 4, 3

    vocab_size = n_community * n_members + 1
    from source.model.model import Model   # your updated Model class

    # ============================================================
    # Build a 3-layer hierarchical predictive + memory model
    # ============================================================
    model = Model(
        total_layers = 3,

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
        tau = 0.5,
        device = device
    )

    model.summary()



# %%
if __name__ == "__main__":
    main()

#%%