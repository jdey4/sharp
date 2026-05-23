# measure_text8_sharp_wake_sleep_walltime.py

from sharp.model.model import Model

import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr

import numpy as np
import os
import zipfile
import urllib.request
import argparse
import time


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
# Helpers
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def build_sharp_model(
    device,
    total_layers=5,
    head_layers=2,
    short_term_memory=4,
    vocab_size=27,
    hidden_size=512,
    embedding_dim=100,
    recon_threshold=1e-2,
    context_tag_buffer_size=20,
):
    model = Model(
        total_layers=total_layers,
        num_layers_prediction_head=head_layers,

        vocab_size=vocab_size,
        hidden_sizes=[hidden_size] * total_layers,
        embedding_dim=embedding_dim,

        lr_layers=1e-4,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={
            "weight_decay": 1e-12
        },

        short_term_memory=short_term_memory,
        context_tag_buffer_size=context_tag_buffer_size,
        recon_threshold=recon_threshold,
        sleep=False,
        device=device,
    )

    model.reset_model()
    return model


def run_wake_steps(model, loader, device, num_steps, h_=None, timed=False):
    iterator = iter(loader)

    if timed:
        sync_device(device)
        start = time.perf_counter()

    losses = []
    recon_losses = []

    steps_done = 0

    for _ in range(num_steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

        logits, loss, recon_loss, h_ = model.wake_step(x, y, h_)

        if loss is not None:
            if isinstance(loss, torch.Tensor):
                losses.append(loss.detach().item())
            else:
                losses.append(float(loss))

        if recon_loss is not None:
            if isinstance(recon_loss, torch.Tensor):
                recon_losses.append(recon_loss.detach().item())
            else:
                recon_losses.append(float(recon_loss))

        steps_done += 1

    if timed:
        sync_device(device)
        end = time.perf_counter()
        elapsed = end - start
    else:
        elapsed = None

    return {
        "h": h_,
        "steps_done": steps_done,
        "elapsed_sec": elapsed,
        "mean_loss": float(np.mean(losses)) if len(losses) > 0 else None,
        "mean_recon_loss": float(np.mean(recon_losses)) if len(recon_losses) > 0 else None,
    }


def measure_sleep_step(model, device, sleep_steps=1000):
    sync_device(device)
    start = time.perf_counter()

    model.sleep_step(total_steps=sleep_steps)

    sync_device(device)
    end = time.perf_counter()

    elapsed = end - start

    return {
        "sleep_steps": sleep_steps,
        "elapsed_sec": elapsed,
        "walltime_per_1000_sleep_steps_sec": elapsed / sleep_steps * 1000.0,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', 'mps', or 'cpu'")

    parser.add_argument("--num_tokens", type=int, default=50_000,
                        help="Number of text8 tokens to load for timing/prefill.")

    parser.add_argument("--wake_warmup_steps", type=int, default=100)
    parser.add_argument("--wake_measure_steps", type=int, default=1000)

    parser.add_argument("--sleep_prefill_steps", type=int, default=20_000,
                        help="Wake steps before timing sleep, to populate SHARP memory/buffers.")

    parser.add_argument("--sleep_steps", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--short_term_memory", type=int, default=4)

    parser.add_argument("--total_layers", type=int, default=5)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--recon_threshold", type=float, default=1e-2)
    parser.add_argument("--context_tag_buffer_size", type=int, default=20)

    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)

    # -----------------------------
    # Load text8
    # -----------------------------
    text = download_text8()
    stoi, _ = build_vocab(text)
    encoded = encode(text, stoi)

    required_tokens = max(
        args.num_tokens,
        args.sleep_prefill_steps + args.wake_measure_steps + args.short_term_memory + 10
    )

    encoded_small = encoded[:required_tokens]

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
    print("Dataset length:", len(dataset))
    print("Batch size:", args.batch_size)
    print("Short-term memory:", args.short_term_memory)

    # -----------------------------
    # Build SHARP
    # -----------------------------
    model = build_sharp_model(
        device=device,
        total_layers=args.total_layers,
        head_layers=args.head_layers,
        short_term_memory=args.short_term_memory,
        vocab_size=len(stoi),
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        recon_threshold=args.recon_threshold,
        context_tag_buffer_size=args.context_tag_buffer_size,
    )

    print("SHARP parameters:", f"{count_parameters(model):,}")

    # ========================================================
    # 1. Wake timing
    # ========================================================
    print("\n" + "=" * 80)
    print("Measuring SHARP wake wall-time")
    print("=" * 80)

    h_ = None

    warmup_result = run_wake_steps(
        model=model,
        loader=loader,
        device=device,
        num_steps=args.wake_warmup_steps,
        h_=h_,
        timed=False,
    )

    h_ = warmup_result["h"]

    wake_result = run_wake_steps(
        model=model,
        loader=loader,
        device=device,
        num_steps=args.wake_measure_steps,
        h_=h_,
        timed=True,
    )

    wake_elapsed = wake_result["elapsed_sec"]
    wake_tokens = args.wake_measure_steps * args.batch_size
    wake_walltime_per_1000 = wake_elapsed / wake_tokens * 1000.0

    print(f"Wake measured online tokens: {wake_tokens}")
    print(f"Wake total timed wall-time: {wake_elapsed:.6f} s")
    print(f"Wake wall-time / 1000 tokens (s): {wake_walltime_per_1000:.2f}")
    print(f"Wake mean loss: {wake_result['mean_loss']}")
    print(f"Wake mean recon loss: {wake_result['mean_recon_loss']}")

    # ========================================================
    # 2. Sleep timing
    # ========================================================
    print("\n" + "=" * 80)
    print("Preparing SHARP memory before sleep timing")
    print("=" * 80)

    # Rebuild model for clean sleep timing prefill.
    model_sleep = build_sharp_model(
        device=device,
        total_layers=args.total_layers,
        head_layers=args.head_layers,
        short_term_memory=args.short_term_memory,
        vocab_size=len(stoi),
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        recon_threshold=args.recon_threshold,
        context_tag_buffer_size=args.context_tag_buffer_size,
    )

    h_sleep = None

    prefill_result = run_wake_steps(
        model=model_sleep,
        loader=loader,
        device=device,
        num_steps=args.sleep_prefill_steps,
        h_=h_sleep,
        timed=True,
    )

    h_sleep = prefill_result["h"]

    print(f"Sleep prefill wake steps: {args.sleep_prefill_steps}")
    print(f"Sleep prefill wall-time: {prefill_result['elapsed_sec']:.6f} s")
    print(f"Sleep prefill mean loss: {prefill_result['mean_loss']}")
    print(f"Sleep prefill mean recon loss: {prefill_result['mean_recon_loss']}")

    print("\n" + "=" * 80)
    print("Measuring SHARP sleep wall-time")
    print("=" * 80)

    sleep_result = measure_sleep_step(
        model=model_sleep,
        device=device,
        sleep_steps=args.sleep_steps
    )

    print(f"Sleep replay steps: {sleep_result['sleep_steps']}")
    print(f"Sleep total timed wall-time: {sleep_result['elapsed_sec']:.6f} s")
    print(
        f"Sleep wall-time / 1000 replay steps (s): "
        f"{sleep_result['walltime_per_1000_sleep_steps_sec']:.2f}"
    )

    # ========================================================
    # Summary
    # ========================================================
    print("\n" + "=" * 80)
    print("SHARP WAKE/SLEEP WALL-TIME SUMMARY")
    print("=" * 80)

    print(f"Parameters: {count_parameters(model):,}")
    print(f"Wake wall-time / 1000 online tokens (s): {wake_walltime_per_1000:.2f}")
    print(
        f"Sleep wall-time / 1000 replay steps (s): "
        f"{sleep_result['walltime_per_1000_sleep_steps_sec']:.2f}"
    )

    print("\nRecommended table entry format:")
    print(
        f"SHARP & ... & "
        f"${wake_walltime_per_1000:.2f}$ wake, "
        f"${sleep_result['walltime_per_1000_sleep_steps_sec']:.2f}$ sleep"
    )


if __name__ == "__main__":
    main()