import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import zipfile
import urllib.request
import argparse
from tqdm import tqdm
from transformer_model import Transformer, CONFIGS


def download_text8(path="dataset/text8.zip"):
    url = "http://mattmahoney.net/dc/text8.zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading text8...")
        urllib.request.urlretrieve(url, path)
    with zipfile.ZipFile(path) as zf:
        return zf.read(zf.namelist()[0]).decode("utf-8")


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)


class SeqDataset(Dataset):
    def __init__(self, data, seq_len):
        n = (len(data) - 1) // seq_len
        self.x = torch.from_numpy(data[:n * seq_len].reshape(n, seq_len).copy()).long()
        self.y = torch.from_numpy(data[1:n * seq_len + 1].reshape(n, seq_len).copy()).long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


parser = argparse.ArgumentParser()
parser.add_argument("--model_no", type=int, default=1)
parser.add_argument("--model_size", type=str, default="10M", choices=list(CONFIGS.keys()))
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

device = args.device

cfg = CONFIGS[args.model_size]
text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

train_sample = 90_000_000
segment = encoded[(args.model_no - 1) * train_sample : args.model_no * train_sample]

dataset = SeqDataset(segment, cfg["max_seq_len"])
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = Transformer(**cfg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-12)

print(f"Transformer {args.model_size} | params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training model_no={args.model_no} on {device}")

model.train()
for x, y in tqdm(loader, desc=f"train m{args.model_no}"):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, cfg["vocab_size"]), y.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

os.makedirs("../saved_models/transformer_baselines", exist_ok=True)
torch.save(
    model.state_dict(),
    f"../saved_models/transformer_baselines/transformer_{args.model_size}_model{args.model_no}_text8_100M.pt",
)
print("Saved.")
