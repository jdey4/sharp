#%%
import os
import re
import math
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset


# ============================================================
# Device
# ============================================================
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

if device == "cuda":
    torch.backends.cudnn.benchmark = True

use_amp = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


# ------------------------------------------------------------
# Choose baseline type here
# ------------------------------------------------------------
model_type = "lstm"   # options: "rnn", "gru", "lstm"


# ------------------------------------------------------------
# Training speed knobs
# ------------------------------------------------------------
stream_batch_size = 256   # number of parallel streams within a book
short_term_memory = 4     # truncated BPTT window length
eval_batch_size = 256     # larger is fine for eval
num_workers = 4 if device == "cuda" else 0


# ============================================================
# Utility
# ============================================================
def compute_bpc(logits, targets):
    """
    logits: (B, V)
    targets: (B,)
    """
    loss = F.cross_entropy(logits, targets, reduction="mean")
    return float(loss.item() / np.log(2.0))


def detach_hidden(h):
    if h is None:
        return None
    if isinstance(h, tuple):  # LSTM
        return tuple(v.detach() for v in h)
    return h.detach()


#%%
# ============================================================
# Step 1: PG-19 loading + preprocessing
# ============================================================
def normalize_to_27_vocab(text):
    """
    Convert raw book text into the same 27-symbol vocabulary as text8:
      - lowercase
      - keep only a-z
      - replace everything else with space
      - collapse repeated spaces
    """
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
    return np.fromiter((stoi[c] for c in text), dtype=np.int64, count=len(text))


def _extract_text_field(example):
    for key in ["text", "book_text", "content", "document", "story"]:
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(f"Could not find text field in keys: {list(example.keys())}")


def load_pg19_books_by_token_budget(
    target_train_chars=100_000_000,
    max_train_chars_per_book=2_000_000,
    max_holdout_books=5,
    min_book_chars=20_000,
    max_eval_chars_per_book=1_000_000,
):
    print("Loading PG-19 from Hugging Face datasets...")
    ds = load_dataset("fla-hub/pg19")

    train_books_raw = []
    total_train_chars = 0

    for ex in tqdm(ds["train"], desc="Collecting train books"):
        raw = _extract_text_field(ex)
        text = normalize_to_27_vocab(raw)

        if len(text) < min_book_chars:
            continue

        text = text[:max_train_chars_per_book]

        if len(text) < min_book_chars:
            continue

        remaining_budget = target_train_chars - total_train_chars
        if remaining_budget <= 0:
            break

        if len(text) > remaining_budget:
            text = text[:remaining_budget].strip()
            if len(text) < min_book_chars:
                break

        train_books_raw.append(text)
        total_train_chars += len(text)

        if len(train_books_raw) % 10 == 0:
            print(
                f"Collected {len(train_books_raw)} books | "
                f"total normalized chars = {total_train_chars:,}"
            )

        if total_train_chars >= target_train_chars:
            break

    holdout_split = "validation" if "validation" in ds else "test"
    holdout_books_raw = []

    for ex in tqdm(ds[holdout_split], desc=f"Collecting {holdout_split} books"):
        raw = _extract_text_field(ex)
        text = normalize_to_27_vocab(raw)

        if len(text) < min_book_chars:
            continue

        text = text[:max_eval_chars_per_book]
        holdout_books_raw.append(text)

        if len(holdout_books_raw) >= max_holdout_books:
            break

    print(f"\nFinal training book count: {len(train_books_raw)}")
    print(f"Total normalized training chars: {total_train_chars:,}")
    print(f"Max train chars per book: {max_train_chars_per_book:,}")
    print(f"Holdout books: {len(holdout_books_raw)} from split='{holdout_split}'")

    return train_books_raw, holdout_books_raw, total_train_chars


#%%
# ============================================================
# Step 2: Stream batching helpers
# ============================================================
def make_book_stream_tensors(encoded_book, batch_size):
    """
    Turn one long encoded book into B parallel streams.

    Returns:
        data:    (B, S)
        targets: (B, S)
    where each row is one contiguous stream.
    """
    encoded_book = np.asarray(encoded_book, dtype=np.int64)

    usable = (len(encoded_book) - 1) // batch_size
    if usable <= 1:
        return None, None

    total_tokens = usable * batch_size + 1
    arr = encoded_book[:total_tokens]

    x = arr[:-1].reshape(batch_size, usable)
    y = arr[1:].reshape(batch_size, usable)

    return x, y


def iter_stream_windows(x_streams, y_streams, seq_len):
    """
    Yield contiguous windows from stream-batched tensors.

    x_streams, y_streams: (B, S)
    Returns x, y where:
        x: (B, T)
        y: (B,)
    using target at the final step of each window.
    """
    B, S = x_streams.shape
    if S <= seq_len:
        return

    for start in range(0, S - seq_len):
        x = x_streams[:, start:start + seq_len]
        y = y_streams[:, start + seq_len - 1]
        yield x, y


#%%
# ============================================================
# Step 3: Baseline model
# ============================================================
class CharRNNBaseline(nn.Module):
    def __init__(
        self,
        vocab_size=27,
        embedding_dim=100,
        hidden_size=512,
        num_layers=5,
        model_type="rnn",
        device="cpu",
    ):
        super().__init__()

        self.model_type = model_type.lower()
        self.device = torch.device(device)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.model_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError("model_type must be one of: 'rnn', 'gru', 'lstm'")

        self.readout = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        """
        x: (B, T)
        h:
          - RNN/GRU: (num_layers, B, H)
          - LSTM: ((num_layers, B, H), (num_layers, B, H))
        """
        x = x.to(self.device, non_blocking=True)
        emb = self.embedding(x)
        out, h = self.rnn(emb, h)
        logits = self.readout(out[:, -1, :])
        return logits, h


#%%
# ============================================================
# Step 4: Evaluation helper
# ============================================================
@torch.no_grad()
def evaluate_books(
    model,
    books_encoded,
    short_term_memory=4,
    max_tokens_per_book=None,
    batch_size=256,
):
    total_bpc = 0.0
    total_correct = 0
    total_count = 0

    model.eval()

    for encoded_book in books_encoded:
        if max_tokens_per_book is not None:
            encoded_book = encoded_book[:max_tokens_per_book]

        if len(encoded_book) <= short_term_memory + 1:
            continue

        x_streams, y_streams = make_book_stream_tensors(encoded_book, batch_size)
        if x_streams is None:
            continue

        x_streams = torch.from_numpy(x_streams).to(model.device, non_blocking=True)
        y_streams = torch.from_numpy(y_streams).to(model.device, non_blocking=True)

        h = None

        for x, y in iter_stream_windows(x_streams, y_streams, short_term_memory):
            logits, h = model(x)
            # h = detach_hidden(h)

            total_bpc += compute_bpc(logits, y)
            pred_tok = logits.argmax(dim=-1)
            total_correct += int((pred_tok == y).sum().item())
            total_count += int(y.numel())

    avg_bpc = total_bpc / max(total_count / batch_size, 1e-8)
    avg_acc = total_correct / max(total_count, 1)
    return avg_bpc, avg_acc


#%%
# ============================================================
# Step 5: Load PG-19 subset by total token budget
# ============================================================
stoi, itos = build_fixed_27_vocab()

target_train_chars = 100_000_000
max_train_chars_per_book = 2_000_000
max_holdout_books = 5
min_book_chars = 20_000
max_eval_chars_per_book = 1_000_000

train_books_raw, holdout_books_raw, total_train_chars = load_pg19_books_by_token_budget(
    target_train_chars=target_train_chars,
    max_train_chars_per_book=max_train_chars_per_book,
    max_holdout_books=max_holdout_books,
    min_book_chars=min_book_chars,
    max_eval_chars_per_book=max_eval_chars_per_book,
)

train_books_encoded = [encode(book, stoi) for book in train_books_raw]
holdout_books_encoded = [encode(book, stoi) for book in holdout_books_raw]

print("Number of training books:", len(train_books_encoded))
print("Number of holdout books:", len(holdout_books_encoded))
print("First 5 train book lengths:", [len(x) for x in train_books_encoded[:5]])


#%%
# ============================================================
# Step 6: Build baseline model
# ============================================================
model_no = 1
vocab_size = 27

embedding_dim = 100
hidden_size = 512
num_layers = 5
lr = 1e-4
weight_decay = 1e-12
grad_clip = 1.0

model = CharRNNBaseline(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    model_type=model_type,
    device=device,
).to(device)

if device == "cuda":
    try:
        model = torch.compile(model)
        print("Using torch.compile")
    except Exception as e:
        print("torch.compile unavailable:", e)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)

criterion = nn.CrossEntropyLoss()
print(model)


#%%
# ============================================================
# Step 7: Training loop
# ============================================================
print(f"\nTraining {model_type.upper()} baseline on PG-19 subset with {target_train_chars:,} normalized chars")
print(f"Each training book capped at {max_train_chars_per_book:,} chars")
print(f"Stream batch size: {stream_batch_size}")
print(f"Short-term memory: {short_term_memory}")

model.train()

ii = 0
chars_seen = 0
correct_ring = np.zeros(1000, dtype=np.float32)
bpc_ring = np.zeros(1000, dtype=np.float32)

for rep in range(1):
    for book_idx, encoded_book in enumerate(train_books_encoded):
        print(
            f"\n=== Training on book {book_idx + 1}/{len(train_books_encoded)} "
            f"| chars={len(encoded_book):,} ==="
        )

        x_streams, y_streams = make_book_stream_tensors(encoded_book, stream_batch_size)
        if x_streams is None:
            print("Skipped: book too short for chosen stream batch size")
            continue

        x_streams = torch.from_numpy(x_streams).to(device, non_blocking=True)
        y_streams = torch.from_numpy(y_streams).to(device, non_blocking=True)

        h = None

        total_steps = x_streams.shape[1] - short_term_memory
        pbar = tqdm(
            iter_stream_windows(x_streams, y_streams, short_term_memory),
            total=total_steps,
            desc=f"Book {book_idx + 1}",
        )

        for x, y in pbar:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits, h = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            h = detach_hidden(h)

            with torch.no_grad():
                ii += 1
                chars_seen += int(y.numel())

                ring_idx = ii % 1000
                bpc_ring[ring_idx] = compute_bpc(logits, y)
                pred_tok = logits.argmax(dim=-1)
                correct_ring[ring_idx] = float((pred_tok == y).float().mean().item())

                if ii % 1000 == 0:
                    acc = float(np.mean(correct_ring))
                    bpc = float(np.mean(bpc_ring))
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{acc:.4f}",
                        bpc=f"{bpc:.4f}",
                        chars_seen=f"{chars_seen:,}",
                    )


#%%
# ============================================================
# Step 8: Save model
# ============================================================
os.makedirs("../saved_models/pg19_models", exist_ok=True)
model_save_path = f"../saved_models/pg19_models/{model_type}_model{model_no}_pg19_100M_cap2M_streambs{stream_batch_size}.pt"

raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
torch.save(raw_model.state_dict(), model_save_path)
print("Saved model to:", model_save_path)


#%%
# ============================================================
# Step 9: Evaluation
# ============================================================
num_backward_books = min(5, len(train_books_encoded))
num_current_books = min(5, len(train_books_encoded))

backward_books = train_books_encoded[:num_backward_books]
current_books = train_books_encoded[-num_current_books:]
forward_books = holdout_books_encoded

forward_bpc, forward_acc = evaluate_books(
    raw_model,
    forward_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_chars_per_book,
    batch_size=eval_batch_size,
)

backward_bpc, backward_acc = evaluate_books(
    raw_model,
    backward_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_chars_per_book,
    batch_size=eval_batch_size,
)

current_bpc, current_acc = evaluate_books(
    raw_model,
    current_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_chars_per_book,
    batch_size=eval_batch_size,
)

print("\n================ FINAL EVALUATION ================")
print(f"Model type: {model_type.upper()}")
print(f"Forward  | BPC: {forward_bpc:.6f} | Acc: {forward_acc:.6f}")
print(f"Backward | BPC: {backward_bpc:.6f} | Acc: {backward_acc:.6f}")
print(f"Current  | BPC: {current_bpc:.6f} | Acc: {current_acc:.6f}")
print("=================================================\n")


#%%
# ============================================================
# Step 10: Save summary
# ============================================================
summary = {
    "model_type": model_type,
    "forward_bpc": forward_bpc,
    "forward_acc": forward_acc,
    "backward_bpc": backward_bpc,
    "backward_acc": backward_acc,
    "current_bpc": current_bpc,
    "current_acc": current_acc,
    "num_train_books": len(train_books_encoded),
    "num_holdout_books": len(holdout_books_encoded),
    "target_train_chars": target_train_chars,
    "actual_train_chars": total_train_chars,
    "max_train_chars_per_book": max_train_chars_per_book,
    "max_eval_chars_per_book": max_eval_chars_per_book,
    "stream_batch_size": stream_batch_size,
    "short_term_memory": short_term_memory,
    "device": device,
}

os.makedirs("../pickle_files", exist_ok=True)
summary_path = f"../pickle_files/result_pg19_{model_type}_100M_cap2M_streambs{stream_batch_size}.pickle"
with open(summary_path, "wb") as handle:
    pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved evaluation summary to {summary_path}")
# %%