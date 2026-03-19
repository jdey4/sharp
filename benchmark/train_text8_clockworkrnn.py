#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import os
import zipfile
import urllib.request
from tqdm import tqdm
import math
import argparse

#%%
device = "mps"  # or: torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

#%%
# ============================================================
# Utilities
# ============================================================
def download_text8(path="dataset/text8.zip"):
    url = "http://mattmahoney.net/dc/text8.zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading text8...")
        urllib.request.urlretrieve(url, path)
    with zipfile.ZipFile(path) as zf:
        data = zf.read(zf.namelist()[0]).decode("utf-8")
    return data

def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)

def compute_bpc(logits, targets):
    """
    logits: (B, V)
    targets: (B,)
    """
    loss_nats = F.cross_entropy(logits, targets, reduction="mean")
    return loss_nats.item() / math.log(2)

class Dataset_converter(Dataset):
    def __init__(self, encoded_text, working_memory=1, short_term_memory=8):
        self.X = []
        self.y = []

        for ii in range(0, len(encoded_text) - working_memory - short_term_memory, 1):
            self.X.append(encoded_text[ii:ii + short_term_memory])
            self.y.append(encoded_text[ii + short_term_memory])

        self.X = tnsr(np.array(self.X)).long()
        self.y = tnsr(np.array(self.y)).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]

#%%
# ============================================================
# Clockwork RNN
# ============================================================
class ClockworkRNN(nn.Module):
    """
    CW-RNN with:
      - 5 modules
      - hidden size 512 per module
      - standard periods [1, 2, 4, 8, 16]

    Connectivity:
      Slower modules can receive from faster/equal modules.
      If modules are ordered from fast -> slow, then module i
      receives recurrent input from modules 0...i.
    """
    def __init__(
        self,
        vocab_size=27,
        embedding_dim=100,
        module_hidden_size=512,
        periods=(1, 2, 4, 8, 16),
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.module_hidden_size = module_hidden_size
        self.periods = list(periods)
        self.num_modules = len(self.periods)
        self.total_hidden_size = self.num_modules * self.module_hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Each module has:
        # 1) input projection from embedding
        # 2) recurrent projection from concatenated faster/equal modules
        self.in_linears = nn.ModuleList()
        self.rec_linears = nn.ModuleList()

        for i in range(self.num_modules):
            self.in_linears.append(
                nn.Linear(embedding_dim, module_hidden_size)
            )

            # recurrent input comes from modules [0, ..., i]
            rec_in_dim = (i + 1) * module_hidden_size
            self.rec_linears.append(
                nn.Linear(rec_in_dim, module_hidden_size, bias=False)
            )

        self.readout = nn.Linear(self.total_hidden_size, vocab_size)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_hidden(self, batch_size, device):
        """
        Returns hidden state as a list of tensors:
            [h_0, h_1, ..., h_{M-1}]
        each of shape (B, Hm)
        """
        return [
            torch.zeros(batch_size, self.module_hidden_size, device=device)
            for _ in range(self.num_modules)
        ]

    def detach_hidden(self, h):
        if h is None:
            return None
        return [hi.detach() for hi in h]

    def forward(self, x, h=None, start_t=0):
        """
        x: (B, T)
        h: list of hidden states, each (B, Hm)
        start_t: global clock start for this forward pass

        Returns:
            logits: (B, vocab_size)
            h: updated hidden list
            end_t: updated global time after consuming T steps
        """
        B, T = x.shape
        emb = self.embedding(x)  # (B, T, E)

        if h is None:
            h = self.init_hidden(B, x.device)

        for s in range(T):
            current_t = start_t + s
            x_t = emb[:, s, :]  # (B, E)

            old_h = h
            new_h = []

            for i in range(self.num_modules):
                if current_t % self.periods[i] == 0:
                    rec_input = torch.cat(old_h[:i+1], dim=-1)  # faster/equal modules
                    h_i = torch.tanh(
                        self.in_linears[i](x_t) + self.rec_linears[i](rec_input)
                    )
                else:
                    h_i = old_h[i]

                new_h.append(h_i)

            h = new_h

        h_cat = torch.cat(h, dim=-1)     # (B, total_hidden_size)
        logits = self.readout(h_cat)     # (B, vocab_size)
        end_t = start_t + T

        return logits, h, end_t

#%%
# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_no",
    type=int,
    default=1,
    help="Segment index of text8 to train on"
)

args = parser.parse_args()

model_no = args.model_no
print("Running model_no:", model_no)

# ---- Parameters ----
short_term_memory = 4
embedding_dim = 100
vocab_size = 27

num_modules = 5
module_hidden_size = 512
periods = [1, 2, 4, 8, 16]

text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

train_sample = 10_000_000
test_sample = 10_000_000

train_data_set = Dataset_converter(
    encoded[(model_no - 1) * train_sample : model_no * train_sample],
    short_term_memory=short_term_memory
)

loader = DataLoader(train_data_set, batch_size=1, shuffle=False)

#%%
# ============================================================
# Training function
# ============================================================
def run_experiment(save_dir="../saved_models/baselines_clockwork"):
    os.makedirs(save_dir, exist_ok=True)

    model = ClockworkRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        module_hidden_size=module_hidden_size,
        periods=periods,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-12)

    print(f"\nTraining CW-RNN model {model_no}")
    print(model)
    print(f"Periods: {periods}")
    print(f"Num modules: {num_modules}")
    print(f"Hidden size per module: {module_hidden_size}")
    print(f"Total hidden size: {num_modules * module_hidden_size}")

    ii = 0
    global_t = 0
    h_ = None

    correct_ring = np.zeros(1000)
    bpc_train = np.zeros(1000)

    model.train()

    for epoch in range(1):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, h_, global_t = model(x, h_, start_t=global_t)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            h_ = model.detach_hidden(h_)

            with torch.no_grad():
                ii += 1
                bpc_train[ii % 1000] = compute_bpc(logits, y)
                pred_tok = logits.argmax(dim=-1)
                correct_ring[ii % 1000] = (pred_tok[0] == y[0]).item()

                if ii % 1000 == 0:
                    acc = np.sum(correct_ring) / (1000 if ii >= 1000 else ii)
                    bpc = np.sum(bpc_train) / (1000 if ii >= 1000 else ii)

                    print(
                        "Iter", ii,
                        f"loss: {loss.item():.8e}",
                        "Acc:", acc,
                        "BPC:", bpc
                    )

    save_path = os.path.join(save_dir, f"clockwork_model{model_no}_text8.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved to: {save_path}")

#%%
# ============================================================
# Run
# ============================================================
run_experiment()