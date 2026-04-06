#%%
from sharp.utils import DatasetConverter, compute_bpc, evaluate_model
from sharp.model.model import Model

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

# %%
text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)
train_sample = 90000000
test_sample = 10000000
short_term_memory = 4
total_layers, head_layers, short_term_memory = 5, 2, 4

vocab_size = 27

test_data_set_forward = Dataset_converter(encoded[-1000000:], 1, short_term_memory=short_term_memory)
#%%
total_model = 1
acc_forward = []
bpc_forward = []
acc_backward = []
bpc_backward = []
acc_current = []
bpc_current = []

for model_no in range(1, total_model+1):
    print("Evaluating model ",  model_no)
    test_data_set_backward = Dataset_converter(encoded[(model_no-1)*train_sample:(model_no-1)*train_sample+1000000], short_term_memory=short_term_memory)
    test_data_set_current = Dataset_converter(encoded[model_no*train_sample-1000000:model_no*train_sample], short_term_memory=short_term_memory)

    model = Model(    
            total_layers = total_layers,
            num_layers_prediction_head = head_layers,

            # ---- Layer sizes ----
            vocab_size = vocab_size,                  # layer 0 input dimension
            hidden_sizes = [128, 128, 128, 128, 128],    # H0, H1, H2
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
    model.load_state_dict(torch.load("/Users/jd/sharp/saved_models/model"+str(model_no)+"_text8_100M.pt"))
    avg_acc, avg_bpc = evaluate_model(model, test_data_set_forward, device=device)

    print("Forward accuracy ", avg_acc)
    print("Forward BPC ", avg_bpc)
    acc_forward.append(avg_acc)
    bpc_forward.append(avg_bpc)

    avg_acc, avg_bpc = evaluate_model(model, test_data_set_backward, device=device)

    print("Backward accuracy ", avg_acc)
    print("Backward BPC ", avg_bpc)
    acc_backward.append(avg_acc)
    bpc_backward.append(avg_bpc)

    avg_acc, avg_bpc = evaluate_model(model, test_data_set_current, device=device)

    print("Current accuracy ", avg_acc)
    print("Current BPC ", avg_bpc)
    acc_current.append(avg_acc)
    bpc_current.append(avg_bpc)

print("Average forward accuracy ", np.mean(acc_forward), "+- ", np.std(acc_forward, ddof=1))
print("Average forward BPC ", np.mean(bpc_forward), "+- ", np.std(bpc_forward, ddof=1))
print("Average backward accuracy ", np.mean(acc_backward), "+- ", np.std(acc_backward, ddof=1))
print("Average backward BPC ", np.mean(bpc_backward), "+- ", np.std(bpc_forward, ddof=1))
print("Average current accuracy ", np.mean(acc_current), "+- ", np.std(acc_current, ddof=1))
print("Average current BPC ", np.mean(bpc_current), "+- ", np.std(bpc_current, ddof=1))

summary = (acc_forward, bpc_forward, acc_backward, bpc_backward, acc_current, bpc_current)

with open("/Users/jd/sharp/pickle_files/text8_res_100M.pickle", 'wb') as f:
    pickle.dump(summary, f)

#%%