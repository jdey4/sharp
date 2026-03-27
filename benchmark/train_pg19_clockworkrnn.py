#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
from tqdm import tqdm
import pickle
import math

from datasets import load_dataset

#%%
device = "mps"   # change to "cuda" or "mps" if needed
print("Using device:", device)

#%%
# ============================================================
# Utilities
# ============================================================

def compute_bpc(logits, targets):
    """
    logits: (B, V)
    targets: (B,)
    """
    loss_nats = F.cross_entropy(logits, targets, reduction="mean")
    return loss_nats.item() / math.log(2)


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
    return np.fromiter((stoi[c] for c in text), dtype=np.int32, count=len(text))


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
# Dataset
# ============================================================

class PG19SequenceDataset(Dataset):
    """
    Memory-efficient next-token dataset.
    Stores only the encoded book and slices windows on demand.
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
# Clockwork RNN
# ============================================================

class ClockworkRNN(nn.Module):
    """
    CW-RNN where module i updates only when t % period_i == 0.

    Connectivity:
      If modules are ordered from fast -> slow, module i receives
      recurrent input from modules [0, ..., i] (faster/equal modules).
    """
    def __init__(
        self,
        vocab_size=27,
        embedding_dim=100,
        module_hidden_size=128,
        periods=(1, 2, 4, 8, 16),
        device="cpu",
    ):
        super().__init__()

        self.device = torch.device(device)
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
        h: list of hidden states, one tensor per module, shape (B, H)
        start_t: global clock position for the first token in this window

        Returns:
            logits: (B, V)
            h: updated hidden list
            end_t: start_t + T
        """
        x = x.to(self.device)
        B, T = x.shape
        emb = self.embedding(x)  # (B, T, E)

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


#%%
# ============================================================
# Evaluation helper
# ============================================================

def reset_hidden_state(model):
    return None


@torch.no_grad()
def evaluate_books(model, books_encoded, short_term_memory=4, max_tokens_per_book=None):
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
        global_t = 0

        for x, y in loader:
            x = x.to(model.device)
            y = y.to(model.device)

            logits, h, global_t = model(x, h, start_t=global_t)
            bpc = compute_bpc(logits, y)

            if h is not None:
                h = model.detach_hidden(h)

            pred_tok = logits.argmax(dim=-1)
            total_correct += (pred_tok[0] == y[0]).item()
            total_bpc += bpc
            total_count += 1

    avg_bpc = total_bpc / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_bpc, avg_acc


#%%
# ============================================================
# Load PG-19 subset
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
# Build model
# ============================================================

model_no = 1
short_term_memory = 4
vocab_size = 27

embedding_dim = 100
module_hidden_size = 128
periods = [1, 2, 4, 8, 16]
lr = 1e-4
weight_decay = 1e-12

model = ClockworkRNN(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    module_hidden_size=module_hidden_size,
    periods=periods,
    device=device,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

criterion = nn.CrossEntropyLoss()

print(model)
print(f"Periods: {periods}")
print(f"Num modules: {len(periods)}")
print(f"Hidden size per module: {module_hidden_size}")
print(f"Total hidden size: {len(periods) * module_hidden_size}")

#%%
# ============================================================
# Training loop
# ============================================================

print(f"\nTraining CLOCKWORK RNN baseline on PG-19 subset with {target_train_chars:,} normalized chars")
print(f"Each training book capped at {max_train_chars_per_book:,} chars")

model.train()

ii = 0
chars_seen = 0
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

        # Reset state and time between books
        h = None
        global_t = 0

        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, h, global_t = model(x, h, start_t=global_t)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            h = model.detach_hidden(h)

            # Optional online training stats
            # with torch.no_grad():
            #     ii += 1
            #     chars_seen += 1
            #     ring_idx = ii % 1000
            #     bpc_train[ring_idx] = compute_bpc(logits, y)
            #     pred_tok = logits.argmax(dim=-1)
            #     correct_ring[ring_idx] = (pred_tok[0] == y[0]).item()
            #
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
# Save model
# ============================================================

os.makedirs("../saved_models/pg19_models", exist_ok=True)
torch.save(
    model.state_dict(),
    f"../saved_models/pg19_models/clockwork_model{model_no}_pg19_100M_cap2M_memlite.pt"
)

#%%
# ============================================================
# Evaluation
# ============================================================

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
print("Model type: CLOCKWORK RNN")
print(f"Forward  | BPC: {forward_bpc:.6f} | Acc: {forward_acc:.6f}")
print(f"Backward | BPC: {backward_bpc:.6f} | Acc: {backward_acc:.6f}")
print(f"Current  | BPC: {current_bpc:.6f} | Acc: {current_acc:.6f}")
print("=================================================\n")

#%%
# ============================================================
# Save summary
# ============================================================

summary = {
    "model_type": "clockwork_rnn",
    "periods": periods,
    "module_hidden_size": module_hidden_size,
    "total_hidden_size": len(periods) * module_hidden_size,
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
with open("../pickle_files/result_pg19_clockwork_100M_cap2M_memlite.pickle", "wb") as handle:
    pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved evaluation summary to ../pickle_files/result_pg19_clockwork_100M_cap2M_memlite.pickle")
# %%