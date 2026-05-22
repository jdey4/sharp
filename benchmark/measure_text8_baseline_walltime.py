# measure_text8_baseline_walltime.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import os
import zipfile
import urllib.request
import argparse
import time
import math


# ============================================================
# Device
# ============================================================
def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


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


class Dataset_converter(Dataset):
    def __init__(self, encoded_text, working_memory=1, short_term_memory=4):
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


# ============================================================
# Baseline recurrent model
# ============================================================
class CharRNNBaseline(nn.Module):
    def __init__(
        self,
        cell_type="rnn",          # "rnn", "lstm", "gru"
        vocab_size=27,
        embedding_dim=100,
        hidden_size=512,
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
        emb = self.embedding(x)               # (B, T, E)
        out, h = self.rnn(emb, h)             # out: (B, T, H)
        last_out = out[:, -1, :]              # (B, H)
        logits = self.readout(last_out)       # (B, V)
        return logits, h

    def detach_hidden(self, h):
        if h is None:
            return None
        if self.cell_type == "lstm":
            return (h[0].detach(), h[1].detach())
        return h.detach()


# ============================================================
# Timing helpers
# ============================================================
def sync_device(device):
    """
    Needed for accurate wall-time on CUDA/MPS because operations can be asynchronous.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_walltime(
    cell_type,
    loader,
    device,
    vocab_size=27,
    embedding_dim=100,
    hidden_size=512,
    num_layers=5,
    lr=1e-4,
    warmup_iters=20,
    measure_iters=100,
):
    model = CharRNNBaseline(
        cell_type=cell_type,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-12)

    model.train()
    h_ = None

    iterator = iter(loader)

    # -----------------------------
    # Warmup
    # -----------------------------
    for _ in range(warmup_iters):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

        logits, h_ = model(x, h_)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        h_ = model.detach_hidden(h_)

    sync_device(device)

    # -----------------------------
    # Timed iterations
    # -----------------------------
    iter_times = []
    losses = []

    for _ in range(measure_iters):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

        sync_device(device)
        start = time.perf_counter()

        logits, h_ = model(x, h_)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        h_ = model.detach_hidden(h_)

        sync_device(device)
        end = time.perf_counter()

        iter_times.append(end - start)
        losses.append(loss.item())

    iter_times = np.array(iter_times)

    result = {
        "cell_type": cell_type,
        "params": count_parameters(model),
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "mean_sec_per_iter": float(iter_times.mean()),
        "std_sec_per_iter": float(iter_times.std()),
        "median_sec_per_iter": float(np.median(iter_times)),
        "tokens_per_iter": loader.batch_size,  # each iteration predicts one next char per stream
        "mean_sec_per_token": float(iter_times.mean() / loader.batch_size),
        "tokens_per_sec": float(loader.batch_size / iter_times.mean()),
        "mean_loss": float(np.mean(losses)),
    }

    return result


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', 'mps', or 'cpu'")

    parser.add_argument("--cell_type", type=str, default="all",
                        choices=["rnn", "lstm", "gru", "all"])

    parser.add_argument("--num_tokens", type=int, default=10_000,
                        help="Number of text8 characters to load for timing.")

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--measure_iters", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--short_term_memory", type=int, default=4)

    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)

    # Load text8
    text = download_text8()
    stoi, itos = build_vocab(text)
    encoded = encode(text, stoi)

    # Use only first num_tokens chars for timing
    encoded_small = encoded[:args.num_tokens]

    dataset = Dataset_converter(
        encoded_small,
        short_term_memory=args.short_term_memory
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    vocab_size = len(stoi)
    print("Vocab size:", vocab_size)
    print("Timing dataset length:", len(dataset))
    print("Batch size:", args.batch_size)
    print("Short-term memory:", args.short_term_memory)

    if args.cell_type == "all":
        cell_types = ["rnn", "lstm", "gru"]
    else:
        cell_types = [args.cell_type]

    all_results = []

    for cell_type in cell_types:
        print("\n" + "=" * 80)
        print(f"Measuring {cell_type.upper()}")
        print("=" * 80)

        result = measure_walltime(
            cell_type=cell_type,
            loader=loader,
            device=device,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            lr=args.lr,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )

        all_results.append(result)

        print(f"Model: {cell_type.upper()}")
        print(f"Parameters: {result['params']:,}")
        print(f"Mean sec/iter: {result['mean_sec_per_iter']:.8f}")
        print(f"Std sec/iter: {result['std_sec_per_iter']:.8f}")
        print(f"Median sec/iter: {result['median_sec_per_iter']:.8f}")
        print(f"Mean sec/token: {result['mean_sec_per_token']:.8f}")
        print(f"Tokens/sec: {result['tokens_per_sec']:.2f}")
        print(f"Mean loss during timed window: {result['mean_loss']:.6f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"{'Model':<10} "
        f"{'Params':>15} "
        f"{'sec/iter':>15} "
        f"{'sec/token':>15} "
        f"{'tokens/sec':>15}"
    )

    for r in all_results:
        print(
            f"{r['cell_type'].upper():<10} "
            f"{r['params']:>15,} "
            f"{r['mean_sec_per_iter']:>15.8f} "
            f"{r['mean_sec_per_token']:>15.8f} "
            f"{r['tokens_per_sec']:>15.2f}"
        )


if __name__ == "__main__":
    main()