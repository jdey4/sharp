# measure_text8_clockwork_walltime.py

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


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


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
# Clockwork RNN
# ============================================================
class ClockworkRNN(nn.Module):
    """
    CW-RNN with modules ordered from fast to slow.
    Slower modules receive recurrent input from faster/equal modules.
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

        self.in_linears = nn.ModuleList()
        self.rec_linears = nn.ModuleList()

        for i in range(self.num_modules):
            self.in_linears.append(
                nn.Linear(embedding_dim, module_hidden_size)
            )

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
        """
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
                    rec_input = torch.cat(old_h[:i + 1], dim=-1)
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


# ============================================================
# Helpers
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_clockwork_walltime(
    loader,
    device,
    vocab_size=27,
    embedding_dim=100,
    module_hidden_size=512,
    periods=(1, 2, 4, 8, 16),
    lr=1e-4,
    warmup_iters=20,
    measure_iters=100,
):
    model = ClockworkRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        module_hidden_size=module_hidden_size,
        periods=periods,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-12)

    model.train()

    h_ = None
    global_t = 0
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

        logits, h_, global_t = model(x, h_, start_t=global_t)
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

        logits, h_, global_t = model(x, h_, start_t=global_t)
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

    sec_per_iter = float(iter_times.mean())
    sec_per_token = sec_per_iter / loader.batch_size
    sec_per_1000_tokens = sec_per_token * 1000.0

    result = {
        "model": "Clockwork RNN",
        "params": count_parameters(model),
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "mean_sec_per_iter": sec_per_iter,
        "std_sec_per_iter": float(iter_times.std()),
        "median_sec_per_iter": float(np.median(iter_times)),
        "mean_sec_per_token": sec_per_token,
        "walltime_per_1000_tokens_sec": sec_per_1000_tokens,
        "tokens_per_sec": float(loader.batch_size / sec_per_iter),
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

    parser.add_argument("--num_tokens", type=int, default=10_000,
                        help="Number of text8 characters to load for timing.")

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--measure_iters", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--short_term_memory", type=int, default=4)

    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--module_hidden_size", type=int, default=512)

    parser.add_argument("--periods", type=str, default="1,2,4,8,16",
                        help="Comma-separated clock periods, e.g. '1,2,4,8,16'")

    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)

    periods = tuple(int(p.strip()) for p in args.periods.split(","))

    text = download_text8()
    stoi, _ = build_vocab(text)
    encoded = encode(text, stoi)

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

    print("Vocab size:", len(stoi))
    print("Timing dataset length:", len(dataset))
    print("Batch size:", args.batch_size)
    print("Short-term memory:", args.short_term_memory)
    print("Periods:", periods)

    result = measure_clockwork_walltime(
        loader=loader,
        device=device,
        vocab_size=len(stoi),
        embedding_dim=args.embedding_dim,
        module_hidden_size=args.module_hidden_size,
        periods=periods,
        lr=args.lr,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
    )

    print("\n" + "=" * 80)
    print("CLOCKWORK RNN WALL-TIME SUMMARY")
    print("=" * 80)
    print(f"Parameters: {result['params']:,}")
    print(f"Mean sec/iter: {result['mean_sec_per_iter']:.8f}")
    print(f"Std sec/iter: {result['std_sec_per_iter']:.8f}")
    print(f"Median sec/iter: {result['median_sec_per_iter']:.8f}")
    print(f"Mean sec/token: {result['mean_sec_per_token']:.8f}")
    print(f"Wall-time / 1000 tokens (s): {result['walltime_per_1000_tokens_sec']:.2f}")
    print(f"Tokens/sec: {result['tokens_per_sec']:.2f}")
    print(f"Mean loss during timed window: {result['mean_loss']:.6f}")


if __name__ == "__main__":
    main()