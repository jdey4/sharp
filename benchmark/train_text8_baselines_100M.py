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
# Baseline recurrent model
# ============================================================
class CharRNNBaseline(nn.Module):
    def __init__(
        self,
        cell_type="rnn",          # "rnn", "lstm", "gru"
        vocab_size=27,
        embedding_dim=100,
        hidden_size=1024,
        num_layers=5,
    ):
        super().__init__()

        self.cell_type = cell_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity="tanh",
                batch_first=True,
            )
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")

        self.readout = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        """
        x: (B, T)
        """
        emb = self.embedding(x)               # (B, T, E)
        out, h = self.rnn(emb, h)             # out: (B, T, H)

        last_out = out[:, -1, :]              # predict next token from final time step
        logits = self.readout(last_out)       # (B, V)

        return logits, h

    def detach_hidden(self, h):
        if h is None:
            return None
        if self.cell_type == "lstm":
            return (h[0].detach(), h[1].detach())
        return h.detach()

#%%
# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_no",
    type=int,
    default=1,
    help="Segment index of text8 to train on",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help='Torch device, e.g. "cuda:0", "cuda:1", or "cpu"',
)
parser.add_argument(
    "--cell_type",
    type=str,
    default="all",
    choices=["rnn", "lstm", "gru", "all"],
    help="Train one of rnn/lstm/gru, or 'all' to run the three sequentially on this device.",
)

args = parser.parse_args()

device = args.device
model_no = args.model_no
print("Using device:", device)
print("Running model_no:", model_no)
print("cell_type:", args.cell_type)

# ---- Parameters ----
total_layers = 5
short_term_memory = 4
embedding_dim = 100
hidden_size = 512
vocab_size = 27

text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

train_sample = 90_000_000
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
def run_experiment(cell_type, dev, save_dir="../saved_models/baselines"):
    os.makedirs(save_dir, exist_ok=True)

    model = CharRNNBaseline(
        cell_type=cell_type,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=total_layers,
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-12)

    print(f"\nTraining {cell_type.upper()} model {model_no}")
    print(model)

    ii = 0
    h_ = None

    model.train()

    for epoch in range(1):
        for x, y in tqdm(loader, desc=f"{cell_type} m{model_no}"):
            x = x.to(dev)
            y = y.to(dev)

            logits, h_ = model(x, h_)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # detach hidden state for truncated online training
            h_ = model.detach_hidden(h_)

            # with torch.no_grad():
            #     ii += 1
            #     bpc_train[ii % 1000] = compute_bpc(logits, y)
            #     pred_tok = logits.argmax(dim=-1)
            #     correct_ring[ii % 1000] = (pred_tok[0] == y[0]).item()

            #     if ii % 1000 == 0:
            #         acc = np.sum(correct_ring) / (1000 if ii >= 1000 else ii)
            #         bpc = np.sum(bpc_train) / (1000 if ii >= 1000 else ii)

            #         print(
            #             "Iter", ii,
            #             f"loss: {loss.item():.8e}",
            #             "Acc:", acc,
            #             "BPC:", bpc
            #         )

    save_path = os.path.join(save_dir, f"{cell_type}_model{model_no}_text8_100M.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved to: {save_path}")

#%%
# ============================================================
# Run baseline(s)
# ============================================================
_cell_types = ["rnn", "lstm", "gru"] if args.cell_type == "all" else [args.cell_type]
for _ct in _cell_types:
    run_experiment(_ct, device)