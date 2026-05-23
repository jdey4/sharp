# train_pg19_gpt2_recurrent_full_softmax.py

from sharp.utils import compute_bpc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import pickle

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Device
# ============================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


# ============================================================
# Choose baseline
# ============================================================

# Choose one: "rnn", "lstm", "gru"
recurrent_type = "gru"

assert recurrent_type in ["rnn", "lstm", "gru"], (
    "recurrent_type must be one of: 'rnn', 'lstm', 'gru'"
)


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
# Dataset
# ============================================================

class PG19GPT2Dataset(Dataset):
    def __init__(self, token_ids, short_term_memory=4):
        self.token_ids = token_ids
        self.short_term_memory = short_term_memory
        self.n = len(token_ids) - short_term_memory

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, index):
        x_ids = self.token_ids[index:index + self.short_term_memory]
        y_id = self.token_ids[index + self.short_term_memory]

        return (
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(y_id, dtype=torch.long),
        )


# ============================================================
# Recurrent baseline model: RNN / LSTM / GRU
# ============================================================

class GPT2RecurrentFullSoftmax(nn.Module):
    def __init__(
        self,
        recurrent_type="rnn",
        input_dim=768,
        hidden_size=512,
        num_layers=4,
        vocab_size=50257,
        head_layers=4,
        dropout=0.0,
    ):
        super().__init__()

        recurrent_type = recurrent_type.lower()
        assert recurrent_type in ["rnn", "lstm", "gru"]

        self.recurrent_type = recurrent_type
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.head_layers = head_layers

        if recurrent_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif recurrent_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif recurrent_type == "gru":
            self.recurrent = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        head = []
        for _ in range(max(head_layers - 1, 0)):
            head.extend(
                [
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                ]
            )

        head.extend(
            [
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, vocab_size),
            ]
        )

        self.head = nn.Sequential(*head)

        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.recurrent.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

                # Optional but standard-ish LSTM forget-gate bias.
                # PyTorch gate order for LSTM: input, forget, cell, output.
                if self.recurrent_type == "lstm":
                    hidden = self.hidden_size
                    with torch.no_grad():
                        p[hidden:2 * hidden].fill_(1.0)

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_dense, h=None):
        """
        x_dense: [B, T, 768]
        h:
            RNN/GRU:  [num_layers, B, hidden_size] or None
            LSTM:     tuple(h, c), each [num_layers, B, hidden_size], or None
        """
        out, h_next = self.recurrent(x_dense, h)
        z = out[:, -1, :]          # [B, H]
        logits = self.head(z)      # [B, vocab]
        return logits, h_next


# ============================================================
# Hidden-state detach helper
# ============================================================

def detach_hidden(h):
    """
    Works for:
        RNN/GRU hidden state: Tensor
        LSTM hidden state: tuple(h, c)
    """
    if h is None:
        return None

    if isinstance(h, tuple):
        return tuple(v.detach() for v in h)

    return h.detach()


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

        ds = PG19GPT2Dataset(encoded_book, short_term_memory=short_term_memory)

        loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # Reset between books, matching the SHARP script.
        h = None

        for x_ids, y_token in tqdm(
            loader,
            desc=f"Evaluating {name} book {book_idx + 1}/{len(books_encoded)}",
            leave=False,
        ):
            y_token = y_token.view(-1).long().to(device)

            x_dense = ids_to_gpt2_embeddings(x_ids).to(device)

            logits, h = model(x_dense, h)
            h = detach_hidden(h)

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

target_train_tokens = 25_000_000
max_train_tokens_per_book = None
max_holdout_books = 5
min_book_tokens = 1024
max_eval_tokens_per_book = 100_000

total_layers = 4
head_layers = 4
short_term_memory = 4

hidden_size = 512

lr = 1e-4
weight_decay = 1e-12

save_model_path = (
    f"../saved_models/pg19_models/"
    f"model1_pg19_gpt2_{recurrent_type}_fullsoftmax.pt"
)

save_summary_path = (
    f"../pickle_files/"
    f"result_pg19_gpt2_{recurrent_type}_fullsoftmax.pickle"
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
# Build recurrent model
# ============================================================

model = GPT2RecurrentFullSoftmax(
    recurrent_type=recurrent_type,
    input_dim=GPT2_EMBED_DIM,
    hidden_size=hidden_size,
    num_layers=total_layers,
    vocab_size=GPT2_VOCAB_SIZE,
    head_layers=head_layers,
    dropout=0.0,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)

print(f"\n===== {recurrent_type.upper()} Baseline Summary =====")
print("Input: dense GPT-2 embedding windows")
print("Target: next GPT-2 token ID")
print("Output: full softmax over GPT-2 vocabulary")
print("Recurrent type:", recurrent_type)
print("Recurrent layers:", total_layers)
print("Hidden size:", hidden_size)
print("Prediction head layers:", head_layers)
print("Short-term memory:", short_term_memory)
print("Device:", device)
print("Save path:", save_model_path)
print("================================\n")


# ============================================================
# Train only if saved model does not exist
# ============================================================

if os.path.exists(save_model_path):
    print(f"\nFound trained {recurrent_type.upper()} model at: {save_model_path}")
    print("Skipping training and loading model directly for evaluation.")

    state = torch.load(save_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

else:
    print(f"\nNo trained {recurrent_type.upper()} model found. Starting training.")
    print(f"Training {recurrent_type.upper()} on PG-19")
    print("X: dense GPT-2 embedding windows")
    print("Y: next GPT-2 token ID")
    print("Output: full softmax over GPT-2 vocabulary")

    ii = 0
    tokens_seen = 0

    correct_ring = np.zeros(1000, dtype=np.float32)
    bits_ring = np.zeros(1000, dtype=np.float32)

    model.train()

    for rep in range(1):
        for book_idx, encoded_book in enumerate(train_books_encoded):
            print(
                f"\n=== Training {recurrent_type.upper()} on book "
                f"{book_idx + 1}/{len(train_books_encoded)} "
                f"| GPT-2 tokens={len(encoded_book):,} ===",
                flush=True,
            )

            ds = PG19GPT2Dataset(encoded_book, short_term_memory=short_term_memory)

            loader = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

            # Reset hidden state between books, matching SHARP behavior.
            h = None

            for x_ids, y_token in tqdm(loader):
                y_token = y_token.view(-1).long().to(device)

                with torch.no_grad():
                    x_dense = ids_to_gpt2_embeddings(x_ids).to(device)

                logits, h = model(x_dense, h)
                loss = nn.functional.cross_entropy(logits, y_token)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Truncated streaming credit assignment.
                h = detach_hidden(h)

                with torch.no_grad():
                    ii += 1
                    tokens_seen += 1

                    ring_idx = ii % 1000
                    bits_ring[ring_idx] = compute_bpc(logits, y_token)

                    pred_tok = logits.argmax(dim=-1)
                    correct_ring[ring_idx] = (pred_tok[0] == y_token[0]).item()

                    if ii % 1000 == 0:
                        acc = float(np.mean(correct_ring))
                        bits = float(np.mean(bits_ring))

                        print(
                            "Iter", ii,
                            f"prediction loss: {float(loss):.8e}",
                            "Acc:", acc,
                            "Bits/token:", bits,
                            f"| GPT-2 tokens seen: {tokens_seen:,}",
                            flush=True,
                        )

    torch.save(model.state_dict(), save_model_path)
    print(f"\nSaved {recurrent_type.upper()} model to:", save_model_path)


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
print(f"Model    | {recurrent_type.upper()}")
print(f"Forward  | Bits/token: {forward_bits:.6f} | Acc: {forward_acc:.6f}")
print(f"Backward | Bits/token: {backward_bits:.6f} | Acc: {backward_acc:.6f}")
print(f"Current  | Bits/token: {current_bits:.6f} | Acc: {current_acc:.6f}")
print("=================================================\n")


# ============================================================
# Save summary
# ============================================================

summary = {
    "model": recurrent_type.upper(),
    "recurrent_type": recurrent_type,
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
    "hidden_size": hidden_size,
    "total_layers": total_layers,
    "head_layers": head_layers,
    "short_term_memory": short_term_memory,
    "lr": lr,
    "weight_decay": weight_decay,
    "save_model_path": save_model_path,
}

with open(save_summary_path, "wb") as handle:
    pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved evaluation summary to:", save_summary_path)