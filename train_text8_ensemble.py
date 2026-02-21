#%%
from source.utils import DatasetConverter, compute_bpc
from source.model.model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import itertools 
from collections import deque
import os
import zipfile
import urllib.request
from tqdm import tqdm
#%%
device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Using device:", device)

#%%
# Step 1: Download and extract text8
def download_text8(path="dataset/text8.zip"):
    url = "http://mattmahoney.net/dc/text8.zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading text8...")
        urllib.request.urlretrieve(url, path)
    with zipfile.ZipFile(path) as zf:
        data = zf.read(zf.namelist()[0]).decode('utf-8')
    return data

# Step 2: Build character-level vocabulary
def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

# Step 3: Encode text into integer tokens
def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)

class Dataset_converter(Dataset):
    def __init__(self, encoded_text, working_memory=1, short_term_memory=8):
        
        self.X = []
        self.y = []

        for ii in range(0, len(encoded_text) - working_memory - short_term_memory, 1):
            self.X.append(
                encoded_text[ii:ii+short_term_memory]
            )
            self.y.append(
                encoded_text[ii+short_term_memory]
            )

        self.X = tnsr(np.array(self.X)).long()
        self.y = tnsr(np.array(self.y)).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]

#%%
# ---- Parameters ----
total_model, total_layers, short_term_memory = 5, 5, 4

vocab_size = 27

text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)
train_sample = 90000000
test_sample = 10000000
short_term_memory = 4

train_data_set = Dataset_converter(encoded[:train_sample], 1, short_term_memory=short_term_memory)
test_data_set = Dataset_converter(encoded[train_sample:], 1, short_term_memory=short_term_memory)

loader = DataLoader(train_data_set, batch_size=1, shuffle=False)


# ============================================================
# Build a 3-layer hierarchical predictive + memory model
# ============================================================
model = {}

for ii in tqdm(range(total_model)):
    model[ii] = Model(
        total_layers = total_layers,

        # ---- Layer sizes ----
        vocab_size = vocab_size,                  # layer 0 input dimension
        hidden_sizes = [256, 256, 256, 256, 256],    # H0, H1, H2
        embedding_dim = 50,

        # ---- Learning rates per layer ----
        lr_layers = 1e-4,   

        # ---- Optimizer type (user can choose) ----
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = {
            "weight_decay": 1e-12
        },

        # ---- Sleep hyperparameters ----
        short_term_memory = short_term_memory,
        context_tag_buffer_size=20,
        # ---- Misc ----
        recon_threshold = 1e-2,
        device = device
    )

model[0].summary()

#%%
for jj in range(total_model):
    model[jj].reset_model()

ii = 0 
h_ = [None]*total_model
correct_ring = np.zeros(1000)
bpc_train = np.zeros(1000)
model_to_train = 0

for _ in range(1):
    for x, y in loader:
        if ii%20000==0:
            recon_losses = []
            for jj in range(total_model):
                recon_losses.append(model[jj].recon_loss_ema)
            model_to_train = np.argmax(recon_losses)

            print("Making model ", model_to_train, " plastic")


        logits, loss, recon_loss, h_[model_to_train] = model[model_to_train].wake_step(x, y, h_[model_to_train])


        with torch.no_grad():
            for jj in range(total_model):
                if jj != model_to_train:
                    logits_, _, _, h_[jj] = \
                        model[jj].eval_step_no_train(x, y, h_[jj])
                    logits += logits_
                    
            ii += 1
            bpc_train[ii % 1000] = compute_bpc(logits, y)
            pred_tok = logits.argmax(dim=-1)
            correct_ring[ii % 1000] = (pred_tok[0] == y[0]).item()
            
            if ii%1000 == 0:
                acc = np.sum(correct_ring) / (1000 if ii >= 1000 else ii)
                bpc = np.sum(bpc_train) / (1000 if ii >= 1000 else ii)
                print("Iter ", ii, f"prediction loss: {loss:.8e}", f"Memory loss: {recon_loss:.8e}", "Acc: ", acc, "BPC: ", bpc)
                if model[model_to_train].sleeping:
                    print("Sleep on ", model[model_to_train].recon_loss_ema)

        if ii%10000==0:
            model[model_to_train].sleep(total_steps=1025)

# %%
for jj in range(total_model):
    torch.save(model[jj].state_dict(), "/Users/jd/sleep_experiment/saved_models/model"+str(jj+1) + "_text8.pt")

#%%
# model = Model(**config)
# model.load_state_dict(torch.load("model_text8.pt"))
# model.eval()

# for p in model.parameters():
#     p.requires_grad_(False)







# total = 0
# bpc_test = 0
# test_loader = DataLoader(test_data_set, batch_size=1, shuffle=False)

# for X, y in tqdm(test_loader):
#     with torch.no_grad():
#         for layer in range(total_layers):
#             # print(layer)

#             if layer == 0:
#                 logits, h[layer] = model[0].encoder(X, context[0], h[layer]) 

#                 bpc_test += compute_bpc(logits, y)


#         total += 1
        
# print(f'Finall BPC on test set: {bpc_test/total:.4f}')
# %%
