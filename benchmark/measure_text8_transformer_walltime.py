# measure_text8_transformer_overlapping_window_walltime.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import zipfile
import urllib.request
import argparse
import time

from transformer_model import Transformer, CONFIGS


# ============================================================
# Device helpers
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
# text8 utilities
# ============================================================
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


# ============================================================
# Overlapping-window online Transformer dataset
# ============================================================
class OverlappingWindowDataset(Dataset):
    """
    Each sample:
        x = tokens[i : i + context_len]
        y = tokens[i + context_len]

    So each iteration predicts ONE next token from a length-context_len window.
    This matches the online next-token setup used by RNN/LSTM/GRU/Clockwork.
    """
    def __init__(self, data, context_len=1024, num_steps=1000):
        assert len(data) >= context_len + num_steps + 1, (
            f"Need at least context_len + num_steps + 1 tokens. "
            f"Got len(data)={len(data)}, context_len={context_len}, num_steps={num_steps}."
        )

        self.data = torch.from_numpy(data).long()
        self.context_len = context_len
        self.num_steps = num_steps

    def __len__(self):
        return self.num_steps

    def __getitem__(self, i):
        x = self.data[i : i + self.context_len]
        y = self.data[i + self.context_len]
        return x, y


# ============================================================
# Helpers
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_transformer_overlapping_walltime(
    model_size,
    loader,
    device,
    lr=1e-4,
    warmup_iters=10,
    measure_iters=1000,
):
    cfg = CONFIGS[model_size]

    model = Transformer(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-12)

    model.train()
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

        x = x.to(device)  # (B, context_len)
        y = y.to(device)  # (B,)

        logits = model(x)  # expected shape: (B, context_len, vocab_size)

        # Use only final-position logits to predict the next token.
        final_logits = logits[:, -1, :]  # (B, vocab_size)

        loss = F.cross_entropy(final_logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

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

        logits = model(x)
        final_logits = logits[:, -1, :]
        loss = F.cross_entropy(final_logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        sync_device(device)
        end = time.perf_counter()

        iter_times.append(end - start)
        losses.append(loss.item())

    iter_times = np.array(iter_times)

    # Since each iteration predicts one next token per stream.
    tokens_processed = measure_iters * loader.batch_size

    total_time = float(iter_times.sum())
    mean_sec_per_iter = float(iter_times.mean())
    sec_per_token = total_time / tokens_processed
    walltime_per_1000_tokens = sec_per_token * 1000.0

    return {
        "model_size": model_size,
        "params": count_parameters(model),
        "context_len": cfg["max_seq_len"],
        "batch_size": loader.batch_size,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "tokens_processed": tokens_processed,
        "total_timed_sec": total_time,
        "mean_sec_per_iter": mean_sec_per_iter,
        "std_sec_per_iter": float(iter_times.std()),
        "median_sec_per_iter": float(np.median(iter_times)),
        "sec_per_token": sec_per_token,
        "walltime_per_1000_tokens_sec": walltime_per_1000_tokens,
        "tokens_per_sec": tokens_processed / total_time,
        "mean_loss": float(np.mean(losses)),
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', 'mps', or 'cpu'")

    parser.add_argument("--model_size", type=str, default="10M",
                        choices=list(CONFIGS.keys()))

    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of overlapping online prediction steps to time.")

    parser.add_argument("--warmup_iters", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)

    cfg = CONFIGS[args.model_size]
    context_len = cfg["max_seq_len"]

    print("Transformer model size:", args.model_size)
    print("Transformer config:", cfg)
    print("Overlapping context length:", context_len)

    text = download_text8()
    stoi, _ = build_vocab(text)
    encoded = encode(text, stoi)

    # Need enough tokens for context window + timed prediction steps.
    needed_tokens = context_len + args.num_steps + 1
    encoded_small = encoded[:needed_tokens]

    dataset = OverlappingWindowDataset(
        encoded_small,
        context_len=context_len,
        num_steps=args.num_steps
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    result = measure_transformer_overlapping_walltime(
        model_size=args.model_size,
        loader=loader,
        device=device,
        lr=args.lr,
        warmup_iters=args.warmup_iters,
        measure_iters=len(loader),
    )

    print("\n" + "=" * 80)
    print(f"TRANSFORMER {args.model_size} OVERLAPPING-WINDOW WALL-TIME SUMMARY")
    print("=" * 80)
    print(f"Parameters: {result['params']:,}")
    print(f"Context length: {result['context_len']}")
    print(f"Batch size: {result['batch_size']}")
    print(f"Timed online prediction steps: {result['tokens_processed']}")
    print(f"Total timed wall-time: {result['total_timed_sec']:.6f} s")
    print(f"Mean sec/iter: {result['mean_sec_per_iter']:.8f}")
    print(f"Std sec/iter: {result['std_sec_per_iter']:.8f}")
    print(f"Median sec/iter: {result['median_sec_per_iter']:.8f}")
    print(f"Sec/token: {result['sec_per_token']:.8f}")
    print(f"Wall-time / 1000 tokens (s): {result['walltime_per_1000_tokens_sec']:.2f}")
    print(f"Tokens/sec: {result['tokens_per_sec']:.2f}")
    print(f"Mean loss during timed window: {result['mean_loss']:.6f}")


if __name__ == "__main__":
    main()