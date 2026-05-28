# train_pg19_gpt2_embed_full_softmax_transformer.py
#
# Transformer baseline for PG-19 that mirrors the recurrent reference
# `train_pg19_gpt2_embed_full_softmax_baselines.py`:
#   - GPT-2 tokenizer
#   - frozen GPT-2 input embeddings (768-dim)
#   - full softmax over the GPT-2 vocabulary
#
# Outside of those critical changes the transformer configuration and
# training loop match the previous transformer baseline scripts
# (`train_pg19_transformer.py` / `train_text8_transformer.py`):
#   - CONFIGS / Block / RMSNorm from `transformer_model.py`
#   - Adam, lr=1e-4, weight_decay=1e-12
#   - non-overlapping windows of `max_seq_len` for training,
#     full-sequence cross-entropy
#   - sliding-window evaluation (window=short_term_memory) predicting
#     the last position

from sharp.utils import compute_bpc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import pickle
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from transformer_model import Block, RMSNorm, CONFIGS


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="10M", choices=list(CONFIGS.keys()))
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

device = args.device
pin = device.startswith("cuda")
print("Using device:", device)


# ============================================================
# GPT-2 tokenizer + frozen embedding front-end
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2 = AutoModel.from_pretrained("gpt2")

gpt2_embed = gpt2.get_input_embeddings().to(device)
gpt2_embed.eval()

for p in gpt2_embed.parameters():
    p.requires_grad_(False)

GPT2_VOCAB_SIZE = tokenizer.vocab_size
GPT2_EMBED_DIM = gpt2_embed.weight.shape[1]

print("GPT-2 vocab size:", GPT2_VOCAB_SIZE)
print("GPT-2 embedding dim:", GPT2_EMBED_DIM)


@torch.no_grad()
def ids_to_gpt2_embeddings(x_ids):
    """
    x_ids: [B, T] long token IDs
    returns: [B, T, 768] dense GPT-2 vectors
    """
    return gpt2_embed(x_ids.to(device))


# ============================================================
# PG-19 loading/tokenization
# ============================================================

def _extract_text_field(example):
    for key in ["text", "book_text", "content", "document", "story"]:
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(f"Could not find text field in keys: {list(example.keys())}")


def tokenize_gpt2(text):
    return np.array(tokenizer.encode(text), dtype=np.int64)


def load_pg19_books_by_gpt2_token_budget(
    target_train_tokens=25_000_000,
    max_train_tokens_per_book=None,
    max_holdout_books=5,
    min_book_tokens=1024,
    max_eval_tokens_per_book=100_000,
):
    print("Loading PG-19 from Hugging Face datasets...")
    ds = load_dataset("fla-hub/pg19")

    train_books_encoded = []
    total_train_tokens = 0

    for ex in tqdm(ds["train"], desc="Collecting train books"):
        raw = _extract_text_field(ex)
        ids = tokenize_gpt2(raw)

        if len(ids) < min_book_tokens:
            continue

        if max_train_tokens_per_book is not None:
            ids = ids[:max_train_tokens_per_book]

        if len(ids) < min_book_tokens:
            continue

        remaining = target_train_tokens - total_train_tokens
        if remaining <= 0:
            break

        if len(ids) > remaining:
            ids = ids[:remaining]

        train_books_encoded.append(ids)
        total_train_tokens += len(ids)

        if len(train_books_encoded) % 10 == 0:
            print(
                f"Collected {len(train_books_encoded)} books | "
                f"total GPT-2 tokens = {total_train_tokens:,}",
                flush=True,
            )

        if total_train_tokens >= target_train_tokens:
            break

    holdout_split = "validation" if "validation" in ds else "test"
    holdout_books_encoded = []

    for ex in tqdm(ds[holdout_split], desc=f"Collecting {holdout_split} books"):
        raw = _extract_text_field(ex)
        ids = tokenize_gpt2(raw)

        if len(ids) < min_book_tokens:
            continue

        ids = ids[:max_eval_tokens_per_book]
        holdout_books_encoded.append(ids)

        if len(holdout_books_encoded) >= max_holdout_books:
            break

    print("\nFinal training book count:", len(train_books_encoded))
    print("Total GPT-2 training tokens:", f"{total_train_tokens:,}")
    print("Holdout books:", len(holdout_books_encoded), f"from split='{holdout_split}'")

    return train_books_encoded, holdout_books_encoded, total_train_tokens


# ============================================================
# Datasets
# ============================================================

class TrainSeqDataset(Dataset):
    """
    Non-overlapping windows of `seq_len` GPT-2 tokens, full-sequence
    next-token prediction. Mirrors `train_pg19_transformer.py`.
    """
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
    """
    Sliding window of `window` GPT-2 tokens, predict the next token.
    Matches the reference baseline's `PG19GPT2Dataset` (short_term_memory=4).
    """
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


# ============================================================
# Transformer model with frozen GPT-2 embedding front-end
# ============================================================

class GPT2TransformerFullSoftmax(nn.Module):
    """
    Same block stack as `Transformer` in `transformer_model.py`, but the
    learned `nn.Embedding(vocab_size, d_model)` is replaced with a linear
    projection from frozen GPT-2 embeddings (768-dim) to d_model.
    The lm_head emits a full softmax over the GPT-2 vocabulary.
    """
    def __init__(
        self,
        gpt2_embed_dim,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        max_seq_len,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.gpt2_embed_dim = gpt2_embed_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(gpt2_embed_dim, d_model, bias=False)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, max_seq_len) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_dense):
        """
        x_dense: [B, T, gpt2_embed_dim] frozen GPT-2 token embeddings
        returns: [B, T, vocab_size] logits over the GPT-2 vocabulary
        """
        h = self.input_proj(x_dense)
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(self.norm(h))


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_books(
    model,
    books_encoded,
    short_term_memory=4,
    max_tokens_per_book=None,
    name="eval",
):
    total_bits = 0.0
    total_correct = 0
    total_count = 0

    model.eval()

    for book_idx, encoded_book in enumerate(books_encoded):
        if max_tokens_per_book is not None:
            encoded_book = encoded_book[:max_tokens_per_book]

        if len(encoded_book) <= short_term_memory:
            continue

        ds = EvalSlidingDataset(encoded_book, window=short_term_memory)

        loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
        )

        for x_ids, y_token in tqdm(
            loader,
            desc=f"Evaluating {name} book {book_idx + 1}/{len(books_encoded)}",
            leave=False,
        ):
            y_token = y_token.view(-1).long().to(device, non_blocking=pin)

            x_dense = ids_to_gpt2_embeddings(x_ids).to(device, non_blocking=pin)

            logits_seq = model(x_dense)               # [B, T, V]
            logits = logits_seq[:, -1, :]             # predict last position

            bits = compute_bpc(logits, y_token)
            pred_tok = logits.argmax(dim=-1)

            total_correct += (pred_tok[0] == y_token[0]).item()
            total_bits += bits
            total_count += 1

    avg_bits = total_bits / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)

    return avg_bits, avg_acc


# ============================================================
# Settings
# ============================================================

cfg = CONFIGS[args.model_size]

target_train_tokens = 25_000_000
max_train_tokens_per_book = None
max_holdout_books = 5
min_book_tokens = 1024
max_eval_tokens_per_book = 100_000

short_term_memory = 4
train_seq_len = cfg["max_seq_len"]

lr = 1e-4
weight_decay = 1e-12

save_model_path = (
    f"../saved_models/pg19_models/"
    f"model1_pg19_gpt2_transformer_{args.model_size}_fullsoftmax.pt"
)

save_summary_path = (
    f"../pickle_files/"
    f"result_pg19_gpt2_transformer_{args.model_size}_fullsoftmax.pickle"
)

os.makedirs("../saved_models/pg19_models", exist_ok=True)
os.makedirs("../pickle_files", exist_ok=True)


# ============================================================
# Load data
# ============================================================

train_books_encoded, holdout_books_encoded, total_train_tokens = (
    load_pg19_books_by_gpt2_token_budget(
        target_train_tokens=target_train_tokens,
        max_train_tokens_per_book=max_train_tokens_per_book,
        max_holdout_books=max_holdout_books,
        min_book_tokens=min_book_tokens,
        max_eval_tokens_per_book=max_eval_tokens_per_book,
    )
)

print("Number of training books:", len(train_books_encoded))
print("Number of holdout books:", len(holdout_books_encoded))
print("First 5 train book lengths:", [len(x) for x in train_books_encoded[:5]])


# ============================================================
# Build transformer model
# ============================================================

model = GPT2TransformerFullSoftmax(
    gpt2_embed_dim=GPT2_EMBED_DIM,
    vocab_size=GPT2_VOCAB_SIZE,
    d_model=cfg["d_model"],
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    d_ff=cfg["d_ff"],
    max_seq_len=cfg["max_seq_len"],
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n===== Transformer {args.model_size} Baseline Summary =====")
print("Input: dense GPT-2 embedding windows")
print("Target: next GPT-2 token ID")
print("Output: full softmax over GPT-2 vocabulary")
print("Trainable params:", f"{trainable_params:,}")
print("d_model:", cfg["d_model"])
print("n_layers:", cfg["n_layers"])
print("n_heads:", cfg["n_heads"])
print("d_ff:", cfg["d_ff"])
print("max_seq_len (training window):", cfg["max_seq_len"])
print("Short-term memory (eval window):", short_term_memory)
print("Device:", device)
print("Save path:", save_model_path)
print("================================\n")


# ============================================================
# Train only if saved model does not exist
# ============================================================

if os.path.exists(save_model_path):
    print(f"\nFound trained transformer at: {save_model_path}")
    print("Skipping training and loading model directly for evaluation.")

    state = torch.load(save_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

else:
    print(f"\nNo trained transformer found. Starting training.")
    print(f"Training Transformer {args.model_size} on PG-19")
    print("X: dense GPT-2 embedding windows")
    print("Y: next GPT-2 token ID")
    print("Output: full softmax over GPT-2 vocabulary")

    model.train()

    ii = 0
    tokens_seen = 0

    for rep in range(1):
        for book_idx, encoded_book in enumerate(train_books_encoded):
            print(
                f"\n=== Training Transformer on book "
                f"{book_idx + 1}/{len(train_books_encoded)} "
                f"| GPT-2 tokens={len(encoded_book):,} ===",
                flush=True,
            )

            ds = TrainSeqDataset(encoded_book, train_seq_len)
            if len(ds) == 0:
                continue

            loader = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=pin,
            )

            for x_ids, y_ids in tqdm(loader, desc=f"book {book_idx + 1}", leave=False):
                x_ids = x_ids.to(device, non_blocking=pin)
                y_ids = y_ids.to(device, non_blocking=pin)

                with torch.no_grad():
                    x_dense = ids_to_gpt2_embeddings(x_ids)

                logits = model(x_dense)                       # [B, T, V]
                loss = F.cross_entropy(
                    logits.view(-1, GPT2_VOCAB_SIZE),
                    y_ids.view(-1),
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                ii += 1
                tokens_seen += y_ids.numel()

            print(f"  steps so far: {ii} | GPT-2 tokens seen: {tokens_seen:,}", flush=True)

    torch.save(model.state_dict(), save_model_path)
    print(f"\nSaved transformer model to:", save_model_path)


# ============================================================
# Final evaluation
# ============================================================

num_backward_books = min(5, len(train_books_encoded))
num_current_books = min(5, len(train_books_encoded))

backward_books = train_books_encoded[:num_backward_books]
current_books = train_books_encoded[-num_current_books:]
forward_books = holdout_books_encoded

print("\nStarting final evaluation...")

forward_bits, forward_acc = evaluate_books(
    model,
    forward_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_tokens_per_book,
    name="forward",
)

backward_bits, backward_acc = evaluate_books(
    model,
    backward_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_tokens_per_book,
    name="backward",
)

current_bits, current_acc = evaluate_books(
    model,
    current_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_tokens_per_book,
    name="current",
)

print("\n================ FINAL EVALUATION ================")
print(f"Model    | Transformer {args.model_size}")
print(f"Forward  | Bits/token: {forward_bits:.6f} | Acc: {forward_acc:.6f}")
print(f"Backward | Bits/token: {backward_bits:.6f} | Acc: {backward_acc:.6f}")
print(f"Current  | Bits/token: {current_bits:.6f} | Acc: {current_acc:.6f}")
print("=================================================\n")


# ============================================================
# Save summary
# ============================================================

summary = {
    "model": f"Transformer-{args.model_size}",
    "model_size": args.model_size,
    "forward_bits_per_token": forward_bits,
    "forward_acc": forward_acc,
    "backward_bits_per_token": backward_bits,
    "backward_acc": backward_acc,
    "current_bits_per_token": current_bits,
    "current_acc": current_acc,
    "num_train_books": len(train_books_encoded),
    "num_holdout_books": len(holdout_books_encoded),
    "target_train_tokens": target_train_tokens,
    "actual_train_tokens": total_train_tokens,
    "max_train_tokens_per_book": max_train_tokens_per_book,
    "max_eval_tokens_per_book": max_eval_tokens_per_book,
    "gpt2_vocab_size": GPT2_VOCAB_SIZE,
    "gpt2_embedding_dim": GPT2_EMBED_DIM,
    "d_model": cfg["d_model"],
    "n_layers": cfg["n_layers"],
    "n_heads": cfg["n_heads"],
    "d_ff": cfg["d_ff"],
    "max_seq_len": cfg["max_seq_len"],
    "short_term_memory": short_term_memory,
    "lr": lr,
    "weight_decay": weight_decay,
    "save_model_path": save_model_path,
}

with open(save_summary_path, "wb") as handle:
    pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved evaluation summary to:", save_summary_path)
