#%%
# from source.utils import compute_bpc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
from tqdm import tqdm
import pickle

# Hugging Face datasets
# pip install datasets
from datasets import load_dataset
from sharp.utils import compute_bpc

#%%
device = "cpu"  # change to "cuda" if running on GPU server
print("Using device:", device)

# ------------------------------------------------------------
# Choose baseline type here
# ------------------------------------------------------------
model_type = "lstm"   # options: "rnn", "gru", "lstm"

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

    Returns:
        normalized string containing only [a-z ].
    """
    text = text.lower()
    text = re.sub(r"[^a-z]+", " ", text)   # anything not a-z -> space
    text = re.sub(r"\s+", " ", text)       # collapse repeated spaces
    return text.strip()


def build_fixed_27_vocab():
    """
    Fixed vocabulary:
      a-z plus space
    """
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return np.fromiter((stoi[c] for c in text), dtype=np.int32, count=len(text))


def _extract_text_field(example):
    """
    PG-19 mirrors may expose text under slightly different field names.
    """
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
    """
    Load PG-19 and accumulate training books until we reach at least
    `target_train_chars` normalized characters.

    IMPORTANT:
      - each training book is capped at max_train_chars_per_book
      - holdout books are taken from validation (or test)
      - holdout books are optionally truncated for faster evaluation
    """
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
# Step 2: Memory-efficient dataset wrapper
# ============================================================

class PG19SequenceDataset(Dataset):
    """
    Memory-efficient next-token dataset.

    Instead of building all subsequences in memory, this dataset stores only
    the encoded book and slices windows on demand.
    """
    def __init__(self, encoded_text, short_term_memory=8):
        self.encoded_text = encoded_text
        self.short_term_memory = short_term_memory
        self.n = len(encoded_text) - short_term_memory

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, index):
        x = self.encoded_text[index:index + self.short_term_memory]
        y = self.encoded_text[index + self.short_term_memory]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


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
        x = x.to(self.device)
        emb = self.embedding(x)               # (B, T, E)
        out, h = self.rnn(emb, h)            # out: (B, T, H)
        logits = self.readout(out[:, -1, :]) # (B, V)
        return logits, h


#%%
# ============================================================
# Step 4: Evaluation helper
# ============================================================

def reset_hidden_state(model):
    """
    Reset recurrent hidden state between books during evaluation.
    """
    return None


@torch.no_grad()
def evaluate_books(model, books_encoded, short_term_memory=4, max_tokens_per_book=None):
    """
    Evaluate average BPC and accuracy over a list of encoded books.
    Hidden state is passed sequentially within each book.
    """
    total_bpc = 0.0
    total_correct = 0
    total_count = 0

    model.eval()

    for book_idx, encoded_book in enumerate(books_encoded):
        if max_tokens_per_book is not None:
            encoded_book = encoded_book[:max_tokens_per_book]

        if len(encoded_book) <= short_term_memory:
            continue

        ds = PG19SequenceDataset(
            encoded_book,
            short_term_memory=short_term_memory
        )
        loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        h = reset_hidden_state(model)

        for x, y in loader:
            x = x.to(model.device)
            y = y.to(model.device)

            logits, h = model(x, h)
            bpc = compute_bpc(logits, y)

            # detach recurrent state to avoid graph accumulation
            if h is not None:
                if isinstance(h, tuple):  # LSTM
                    h = tuple(v.detach() for v in h)
                else:
                    h = h.detach()

            pred_tok = logits.argmax(dim=-1)
            total_correct += (pred_tok[0] == y[0]).item()
            total_bpc += bpc
            total_count += 1

    avg_bpc = total_bpc / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_bpc, avg_acc


#%%
# ============================================================
# Step 5: Load PG-19 subset by total token budget
# ============================================================

stoi, itos = build_fixed_27_vocab()

# ------------------------------------------------------------
# Main token budget
# ------------------------------------------------------------
target_train_chars = 100_000_000
max_train_chars_per_book = 2_000_000
max_holdout_books = 5
min_book_chars = 20_000

# ------------------------------------------------------------
# Evaluation size control
# ------------------------------------------------------------
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
short_term_memory = 4
vocab_size = 27

embedding_dim = 100
hidden_size = 512
num_layers = 5
lr = 1e-4
weight_decay = 1e-12

#%%
model = CharRNNBaseline(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    model_type=model_type,
    device=device
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

criterion = nn.CrossEntropyLoss()

print(model)

#%%
# ============================================================
# Step 7: Training loop (1 rep only)
# Train sequentially book by book until all selected books are consumed
# ============================================================

print(f"\nTraining {model_type.upper()} baseline on PG-19 subset with {target_train_chars:,} normalized chars")
print(f"Each training book capped at {max_train_chars_per_book:,} chars")

model.train()

ii = 0
chars_seen = 0
h = None
correct_ring = np.zeros(1000, dtype=np.float32)
bpc_train = np.zeros(1000, dtype=np.float32)

for rep in range(1):
    for book_idx, encoded_book in enumerate(train_books_encoded):
        print(
            f"\n=== Training on book {book_idx + 1}/{len(train_books_encoded)} "
            f"| chars={len(encoded_book):,} ==="
        )

        train_data_set = PG19SequenceDataset(
            encoded_book,
            short_term_memory=short_term_memory
        )
        loader = DataLoader(
            train_data_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        # Reset recurrent hidden state between books
        h = None

        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, h = model(x, h)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            # truncate BPTT to one training window
            if h is not None:
                if isinstance(h, tuple):  # LSTM
                    h = tuple(v.detach() for v in h)
                else:
                    h = h.detach()

            # with torch.no_grad():
            #     ii += 1
            #     chars_seen += 1

            #     ring_idx = ii % 1000
            #     bpc_train[ring_idx] = compute_bpc(logits, y)
            #     pred_tok = logits.argmax(dim=-1)
            #     correct_ring[ring_idx] = (pred_tok[0] == y[0]).item()

            #     if ii % 1000 == 0:
            #         acc = float(np.mean(correct_ring))
            #         bpc = float(np.mean(bpc_train))

            #         print(
            #             "Iter", ii,
            #             f"loss: {loss.item():.8e}",
            #             "Acc:", acc,
            #             "BPC:", bpc,
            #             f"| chars seen in training stream: {chars_seen:,}"
            #         )

#%%
# ============================================================
# Step 8: Save model
# ============================================================

os.makedirs("../saved_models/pg19_models", exist_ok=True)
torch.save(
    model.state_dict(),
    f"../saved_models/pg19_models/{model_type}_model{model_no}_pg19_100M_cap2M_memlite.pt"
)

#%%
# ============================================================
# Step 9: Evaluation
#   - Forward BPC: holdout books
#   - Backward BPC: first few training books
#   - Current BPC: last few training books
# ============================================================

model = CharRNNBaseline(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    model_type=model_type,
    device=device,
).to(device)

# 3) load weights
ckpt_path = f"/Users/jd/sharp/saved_models/pg19_models/{model_type}_model{model_no}_pg19_100M_cap2M_memlite.pt"
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()


num_backward_books = min(5, len(train_books_encoded))
num_current_books = min(5, len(train_books_encoded))

backward_books = train_books_encoded[:num_backward_books]
current_books = train_books_encoded[-num_current_books:]
forward_books = holdout_books_encoded

forward_bpc, forward_acc = evaluate_books(
    model,
    forward_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_chars_per_book
)

backward_bpc, backward_acc = evaluate_books(
    model,
    backward_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_chars_per_book
)

current_bpc, current_acc = evaluate_books(
    model,
    current_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_chars_per_book
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
}

os.makedirs("../pickle_files", exist_ok=True)
with open(f"../pickle_files/result_pg19_{model_type}_100M_cap2M_memlite.pickle", "wb") as handle:
    pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved evaluation summary to ../pickle_files/result_pg19_{model_type}_100M_cap2M_memlite.pickle")
# %%
