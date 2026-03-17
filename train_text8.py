#%%
from source.utils import DatasetConverter, compute_bpc, evaluate_model
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
import pickle
import math
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
model_no = 1
# ---- Parameters ----
total_layers, head_layers, short_term_memory = 5, 2, 4

vocab_size = 27

text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)
train_sample = 10000000
test_sample = 10000000
short_term_memory = 4

train_data_set = Dataset_converter(encoded[(model_no-1)*train_sample:model_no*train_sample], short_term_memory=short_term_memory)
#test_data_set = Dataset_converter(encoded[train_sample:], 1, short_term_memory=short_term_memory)

# res_acc = []
# res_bpc = []

loader = DataLoader(train_data_set, batch_size=1, shuffle=False)
#%%

# ============================================================
# Build a 3-layer hierarchical predictive + memory model
# ============================================================
model = Model(
    total_layers = total_layers,
    num_layers_prediction_head = head_layers,

    # ---- Layer sizes ----
    vocab_size = vocab_size,                  # layer 0 input dimension
    hidden_sizes = [512, 512, 512, 512, 512],    # H0, H1, H2
    embedding_dim = 100,

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

model.summary()

#%%
print("Training model ",  model_no)
model.reset_model()

ii = 0 
h_ = None
correct_ring = np.zeros(1000)
bpc_train = np.zeros(1000)

for _ in range(1):
    for x, y in loader:
        #loss, _, _, _, _ = model.layers[0].train_step(x,y)
        logits, loss, recon_loss, h_ = model.wake_step(x, y, h_)


        with torch.no_grad():
            ii += 1
            bpc_train[ii % 1000] = compute_bpc(logits, y)
            pred_tok = logits.argmax(dim=-1)
            correct_ring[ii % 1000] = (pred_tok[0] == y[0]).item()
            
            if ii%1000 == 0:
                acc = np.sum(correct_ring) / (1000 if ii >= 1000 else ii)
                bpc = np.sum(bpc_train) / (1000 if ii >= 1000 else ii)
                # res_acc.append(acc)
                # res_bpc.append(bpc)

                print("Iter ", ii, f"prediction loss: {loss:.8e}", f"Memory loss: {recon_loss:.8e}", "Acc: ", acc, "BPC: ", bpc)
                # if model.sleeping:
                #     print("Sleep on ", model.recon_loss_ema)

        # if ii%20000==0:
        #     model.sleep(total_steps=1025)

# %%
# summary = (res_acc, res_bpc)
# with open('/Users/jd/sleep_experiment/pickle_files/result_text8.pickle', 'wb') as handle:
#     pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
os.makedirs("./saved_models/sleepless_models", exist_ok=True)
torch.save(
    model.state_dict(),
    f"./saved_models/sleepless_models/model{model_no}_text8.pt"
)
#%%
# model = Model(    
#         total_layers = total_layers,
#         num_layers_prediction_head = head_layers,

#         # ---- Layer sizes ----
#         vocab_size = vocab_size,                  # layer 0 input dimension
#         hidden_sizes = [512, 512, 512, 512, 512],    # H0, H1, H2
#         embedding_dim = 50,

#         # ---- Learning rates per layer ----
#         lr_layers = 1e-4,   

#         # ---- Optimizer type (user can choose) ----
#         optimizer_class = torch.optim.Adam,
#         optimizer_kwargs = {
#             "weight_decay": 1e-12
#         },

#         # ---- Sleep hyperparameters ----
#         short_term_memory = short_term_memory,
#         context_tag_buffer_size=20,
#         # ---- Misc ----
#         recon_threshold = 1e-2,
#         device = device
#     )
# model.load_state_dict(torch.load("/Users/jd/sleep_experiment/saved_models/model2_text8.pt"))


# %%
