#%%
from source.utils import compute_bpc
from source.model.model import Model

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
from tqdm import tqdm
import pickle

# Hugging Face datasets
# pip install datasets
from datasets import load_dataset

#%%
device = "cpu"  # change to "cuda" if running on GPU server
print("Using device:", device)

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
# Step 3: Evaluation helper
# ============================================================

def reset_eval_state(model):
    """
    Reset model state before a clean evaluation pass.
    """
    model.wake = False
    model.store_tags = False
    model.step = 0
    model.recon_loss_ema = 0.0
    model.sleeping = False

    for l in range(model.total_layers):
        H = model.hidden_sizes[l]
        model.h_states[l] = torch.zeros(1, H, device=model.device)


@torch.no_grad()
def evaluate_books(model, books_encoded, short_term_memory=4, max_tokens_per_book=None):
    """
    Evaluate average BPC and accuracy over a list of encoded books.
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

        reset_eval_state(model)
        h_ = None

        for x, y in loader:
            x = x.to(model.device)
            y = y.to(model.device)

            logits, pred_loss, recon_loss, h_ = model.eval_step_no_train(x, y, h_)
            bpc = compute_bpc(logits, y)

            pred_tok = logits.argmax(dim=-1)
            total_correct += (pred_tok[0] == y[0]).item()
            total_bpc += bpc
            total_count += 1

    avg_bpc = total_bpc / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_bpc, avg_acc


#%%
# ============================================================
# Step 4: Load PG-19 subset by total token budget
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
# Step 5: Build the model
# ============================================================

model_no = 1

total_layers = 5
head_layers = 2
short_term_memory = 4
vocab_size = 27

model = Model(
    total_layers=total_layers,
    num_layers_prediction_head=head_layers,

    # ---- Layer sizes ----
    vocab_size=vocab_size,
    hidden_sizes=[512, 512, 512, 512, 512],
    embedding_dim=100,

    # ---- Learning rates per layer ----
    lr_layers=1e-4,

    # ---- Optimizer type ----
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs={
        "weight_decay": 1e-12
    },

    # ---- Sleep hyperparameters ----
    short_term_memory=short_term_memory,
    context_tag_buffer_size=20,

    # ---- Misc ----
    recon_threshold=1e-2,
    device=device
)

model.summary()

#%%
# ============================================================
# Step 6: Training loop (1 rep only)
# Train sequentially book by book until all selected books are consumed
# ============================================================

print(f"\nTraining model on PG-19 subset with {target_train_chars:,} normalized chars")
print(f"Each training book capped at {max_train_chars_per_book:,} chars")

model.reset_model()

ii = 0
chars_seen = 0
h_ = None
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

        # Reset streaming hidden state between books
        h_ = None
        model.wake = False

        for x, y in loader:
            x = x.to(model.device)
            y = y.to(model.device)

            logits, loss, recon_loss, h_ = model.wake_step(x, y, h_)

            with torch.no_grad():
                ii += 1
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
            #             f"prediction loss: {loss:.8e}",
            #             f"Memory loss: {recon_loss:.8e}",
            #             "Acc:", acc,
            #             "BPC:", bpc,
            #             f"| chars seen in training stream: {chars_seen:,}"
            #         )

            #         if model.sleeping:
            #             print("Sleep on", model.recon_loss_ema)

            if ii % 20000 == 0:
                model.sleep(total_steps=1025)

#%%
# ============================================================
# Step 7: Save model
# ============================================================

os.makedirs("./saved_models/pg19_models", exist_ok=True)
torch.save(
    model.state_dict(),
    f"./saved_models/pg19_models/model{model_no}_pg19_100M_cap2M_memlite.pt"
)

#%%
# ============================================================
# Step 8: Evaluation
#   - Forward BPC: holdout books
#   - Backward BPC: first few training books
#   - Current BPC: last few training books
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
print(f"Forward  | BPC: {forward_bpc:.6f} | Acc: {forward_acc:.6f}")
print(f"Backward | BPC: {backward_bpc:.6f} | Acc: {backward_acc:.6f}")
print(f"Current  | BPC: {current_bpc:.6f} | Acc: {current_acc:.6f}")
print("=================================================\n")

#%%
# ============================================================
# Step 9: Save summary
# ============================================================

summary = {
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

os.makedirs("./pickle_files", exist_ok=True)
with open("./pickle_files/result_pg19_subset_100M_cap2M_memlite.pickle", "wb") as handle:
    pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved evaluation summary to ./pickle_files/result_pg19_subset_100M_cap2M_memlite.pickle")
# %%