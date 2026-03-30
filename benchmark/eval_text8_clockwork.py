import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import from_numpy as tnsr
import numpy as np
import zipfile
import urllib.request
import pickle
import math
from tqdm import tqdm

#%%
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

model_type = "clockwork"
model_dir = "/Users/jd/sharp/saved_models/baselines_clockwork"
output_pickle = f"/Users/jd/sharp/pickle_files/text8_{model_type}_res.pickle"

#%%
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
class ClockworkRNN(nn.Module):
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

        self.in_linears = nn.ModuleList()
        self.rec_linears = nn.ModuleList()

        for i in range(self.num_modules):
            self.in_linears.append(nn.Linear(embedding_dim, module_hidden_size))
            rec_in_dim = (i + 1) * module_hidden_size
            self.rec_linears.append(nn.Linear(rec_in_dim, module_hidden_size, bias=False))

        self.readout = nn.Linear(self.total_hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        return [
            torch.zeros(batch_size, self.module_hidden_size, device=device)
            for _ in range(self.num_modules)
        ]

    def forward(self, x, h=None, start_t=0):
        B, T = x.shape
        emb = self.embedding(x)

        if h is None:
            h = self.init_hidden(B, x.device)

        for s in range(T):
            current_t = start_t + s
            x_t = emb[:, s, :]
            old_h = h
            new_h = []

            for i in range(self.num_modules):
                if current_t % self.periods[i] == 0:
                    rec_input = torch.cat(old_h[:i+1], dim=-1)
                    h_i = torch.tanh(
                        self.in_linears[i](x_t) + self.rec_linears[i](rec_input)
                    )
                else:
                    h_i = old_h[i]

                new_h.append(h_i)

            h = new_h

        h_cat = torch.cat(h, dim=-1)
        logits = self.readout(h_cat)
        end_t = start_t + T

        return logits, h, end_t

#%%
def evaluate_model_clockwork(model, dataset, device="cpu"):
    model.eval()

    total_correct = 0
    total_bpc = 0.0
    total_count = 0

    h_ = None
    global_t = 0

    with torch.no_grad():
        for x, y in tqdm(dataset):
            x = x.unsqueeze(0).to(device)   # (1, T)
            y = y.unsqueeze(0).to(device)   # (1,)

            logits, h_, global_t = model(x, h_, start_t=global_t)

            pred = logits.argmax(dim=-1)
            total_correct += (pred == y).sum().item()
            total_bpc += compute_bpc(logits, y)
            total_count += y.size(0)

            h_ = [hi.detach() for hi in h_]

    avg_acc = total_correct / total_count
    avg_bpc = total_bpc / total_count
    return avg_acc, avg_bpc

#%%
text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

train_sample = 10_000_000
short_term_memory = 4
vocab_size = 27

test_data_set_forward = Dataset_converter(
    encoded[-1_000_000:], 1, short_term_memory=short_term_memory
)

#%%
files = sorted([
    f for f in os.listdir(model_dir)
    if f.endswith(".pt")
])

print("Found files:")
for f in files:
    print(f)

if len(files) == 0:
    raise ValueError(f"No .pt files found in {model_dir}")

acc_forward = []
bpc_forward = []
acc_backward = []
bpc_backward = []
acc_current = []
bpc_current = []

for f in files:
    print("\nEvaluating", f)

    name = os.path.splitext(f)[0]
    if "model" not in name:
        print("Skipping unrecognized file:", f)
        continue

    model = ClockworkRNN(
        vocab_size=vocab_size,
        embedding_dim=100,
        module_hidden_size=512,
        periods=(1, 2, 4, 8, 16),
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, f), map_location=device))

    model_no = int(name.split("model")[1].split("_")[0])

    test_data_set_backward = Dataset_converter(
        encoded[(model_no - 1) * train_sample : (model_no - 1) * train_sample + 1_000_000],
        short_term_memory=short_term_memory
    )
    test_data_set_current = Dataset_converter(
        encoded[model_no * train_sample - 1_000_000 : model_no * train_sample],
        short_term_memory=short_term_memory
    )

    avg_acc, avg_bpc = evaluate_model_clockwork(model, test_data_set_forward, device=device)
    print("Forward accuracy", avg_acc)
    print("Forward BPC", avg_bpc)
    acc_forward.append(avg_acc)
    bpc_forward.append(avg_bpc)

    avg_acc, avg_bpc = evaluate_model_clockwork(model, test_data_set_backward, device=device)
    print("Backward accuracy", avg_acc)
    print("Backward BPC", avg_bpc)
    acc_backward.append(avg_acc)
    bpc_backward.append(avg_bpc)

    avg_acc, avg_bpc = evaluate_model_clockwork(model, test_data_set_current, device=device)
    print("Current accuracy", avg_acc)
    print("Current BPC", avg_bpc)
    acc_current.append(avg_acc)
    bpc_current.append(avg_bpc)

print("Average forward accuracy", np.mean(acc_forward), "+-", np.std(acc_forward, ddof=1))
print("Average forward BPC", np.mean(bpc_forward), "+-", np.std(bpc_forward, ddof=1))
print("Average backward accuracy", np.mean(acc_backward), "+-", np.std(acc_backward, ddof=1))
print("Average backward BPC", np.mean(bpc_backward), "+-", np.std(bpc_backward, ddof=1))
print("Average current accuracy", np.mean(acc_current), "+-", np.std(acc_current, ddof=1))
print("Average current BPC", np.mean(bpc_current), "+-", np.std(bpc_current, ddof=1))

summary = (acc_forward, bpc_forward, acc_backward, bpc_backward, acc_current, bpc_current)

with open(output_pickle, "wb") as f:
    pickle.dump(summary, f)