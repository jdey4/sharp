import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
import math
import argparse
import pickle
from tqdm import tqdm
from datasets import load_dataset
from transformer_model import Transformer, CONFIGS

short_term_memory = 4


def normalize_to_27_vocab(text):
    text = text.lower()
    text = re.sub(r"[^a-z]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_fixed_27_vocab():
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return np.fromiter((stoi[c] for c in text), dtype=np.int32, count=len(text))


def _extract_text_field(example):
    for key in ["text", "book_text", "content", "document", "story"]:
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(f"No text field in {list(example.keys())}")


def load_pg19_books(
    target_train_chars=100_000_000,
    max_chars_per_book=2_000_000,
    max_holdout_books=5,
    min_book_chars=20_000,
    max_eval_chars=1_000_000,
):
    print("Loading PG-19...")
    ds = load_dataset("fla-hub/pg19")

    train_books = []
    total = 0
    for ex in tqdm(ds["train"], desc="Train books"):
        raw = _extract_text_field(ex)
        text = normalize_to_27_vocab(raw)
        if len(text) < min_book_chars:
            continue
        text = text[:max_chars_per_book]
        remaining = target_train_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].strip()
            if len(text) < min_book_chars:
                break
        train_books.append(text)
        total += len(text)
        if total >= target_train_chars:
            break

    holdout_split = "validation" if "validation" in ds else "test"
    holdout_books = []
    for ex in tqdm(ds[holdout_split], desc=f"{holdout_split} books"):
        raw = _extract_text_field(ex)
        text = normalize_to_27_vocab(raw)
        if len(text) < min_book_chars:
            continue
        holdout_books.append(text[:max_eval_chars])
        if len(holdout_books) >= max_holdout_books:
            break

    print(f"Train: {len(train_books)} books, {total:,} chars")
    print(f"Holdout: {len(holdout_books)} books")
    return train_books, holdout_books


class TrainSeqDataset(Dataset):
    def __init__(self, data, seq_len):
        n = (len(data) - 1) // seq_len
        if n == 0:
            self.x = torch.zeros(0, seq_len, dtype=torch.long)
            self.y = torch.zeros(0, seq_len, dtype=torch.long)
        else:
            self.x = torch.from_numpy(data[:n * seq_len].reshape(n, seq_len).copy()).long()
            self.y = torch.from_numpy(data[1:n * seq_len + 1].reshape(n, seq_len).copy()).long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class EvalSlidingDataset(Dataset):
    def __init__(self, encoded_text, window=4):
        self.encoded = encoded_text
        self.window = window
        self.n = len(encoded_text) - window

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, i):
        x = torch.tensor(self.encoded[i:i + self.window], dtype=torch.long)
        y = torch.tensor(self.encoded[i + self.window], dtype=torch.long)
        return x, y


@torch.no_grad()
def evaluate_books(model, books_encoded, vocab_size, device, pin, max_tokens=None, max_bpc=4.755):
    model.eval()
    total_bpc = 0.0
    total_correct = 0
    total_count = 0

    for enc in books_encoded:
        if max_tokens:
            enc = enc[:max_tokens]
        ds = EvalSlidingDataset(enc, window=short_term_memory)
        if len(ds) == 0:
            continue
        ldr = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=pin)
        for x, y in tqdm(ldr, desc="eval", leave=False):
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            logits = model(x)
            logits = logits[:, -1, :]
            bpc = F.cross_entropy(logits, y).item() / math.log(2)
            if bpc > max_bpc:
                bpc = max_bpc
            total_bpc += bpc
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_count += 1

    avg_bpc = total_bpc / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_bpc, avg_acc


parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="10M", choices=list(CONFIGS.keys()))
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

device = args.device
pin = device.startswith("cuda")

cfg = CONFIGS[args.model_size]
stoi, itos = build_fixed_27_vocab()
train_seq_len = cfg["max_seq_len"]
vocab_size = cfg["vocab_size"]

target_train_chars = 100_000_000
max_eval_chars = 1_000_000

train_books_raw, holdout_books_raw = load_pg19_books(
    target_train_chars=target_train_chars,
    max_eval_chars=max_eval_chars,
)

train_books_encoded = [encode(b, stoi) for b in train_books_raw]
holdout_books_encoded = [encode(b, stoi) for b in holdout_books_raw]

model = Transformer(**cfg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-12)

print(f"Transformer {args.model_size} | params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on {device}")

model.train()
ii = 0
for book_idx, enc in enumerate(train_books_encoded):
    print(f"\nBook {book_idx + 1}/{len(train_books_encoded)} | chars={len(enc):,}")
    ds = TrainSeqDataset(enc, train_seq_len)
    if len(ds) == 0:
        continue
    ldr = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=pin)
    for x, y in tqdm(ldr, desc=f"book {book_idx+1}", leave=False):
        x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ii += 1
    print(f"  steps so far: {ii}")

os.makedirs("../saved_models/transformer_baselines", exist_ok=True)
torch.save(
    model.state_dict(),
    f"../saved_models/transformer_baselines/transformer_{args.model_size}_pg19.pt",
)

num_backward = min(5, len(train_books_encoded))
num_current = min(5, len(train_books_encoded))

forward_bpc, forward_acc = evaluate_books(
    model, holdout_books_encoded, vocab_size, device, pin, max_eval_chars
)
backward_bpc, backward_acc = evaluate_books(
    model, train_books_encoded[:num_backward], vocab_size, device, pin, max_eval_chars
)
current_bpc, current_acc = evaluate_books(
    model, train_books_encoded[-num_current:], vocab_size, device, pin, max_eval_chars
)

print(f"\nForward  | BPC: {forward_bpc:.6f} | Acc: {forward_acc:.6f}")
print(f"Backward | BPC: {backward_bpc:.6f} | Acc: {backward_acc:.6f}")
print(f"Current  | BPC: {current_bpc:.6f} | Acc: {current_acc:.6f}")

summary = {
    "forward_bpc": forward_bpc,
    "forward_acc": forward_acc,
    "backward_bpc": backward_bpc,
    "backward_acc": backward_acc,
    "current_bpc": current_bpc,
    "current_acc": current_acc,
    "model_size": args.model_size,
}

os.makedirs("../pickle_files", exist_ok=True)
with open(f"../pickle_files/result_pg19_transformer_{args.model_size}.pickle", "wb") as f:
    pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to ../pickle_files/result_pg19_transformer_{args.model_size}.pickle")
