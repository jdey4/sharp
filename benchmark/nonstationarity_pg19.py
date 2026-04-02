#%%
from sharp.utils import compute_bpc
from sharp.model.model import Model

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Hugging Face datasets
from datasets import load_dataset

#%%
device = "cpu"
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
# Step 2: JS-divergence utilities
# ============================================================

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    p = p / p.sum()
    q = q / q.sum()

    return np.sum(p * np.log(p / q))


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def compute_window_histograms(token_ids, vocab_size, window_size, stride, num_windows=None):
    token_ids = np.asarray(token_ids)
    n = len(token_ids)

    if n < window_size:
        raise ValueError(f"window_size={window_size} is larger than sequence length={n}")

    starts = list(range(0, n - window_size + 1, stride))
    if num_windows is not None:
        starts = starts[:num_windows]

    histograms = []
    for w_idx, start in enumerate(starts):
        end = start + window_size
        window = token_ids[start:end]

        counts = np.bincount(window, minlength=vocab_size).astype(np.float64)
        probs = counts / counts.sum()

        histograms.append({
            "window_index": w_idx,
            "start": start,
            "end": end,
            "probs": probs,
        })

    return histograms


def compute_pairwise_js_matrix(histograms):
    n = len(histograms)
    mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            mat[i, j] = js_divergence(histograms[i]["probs"], histograms[j]["probs"])

    return mat


#%%
# ============================================================
# Step 3: Plot utility
# ============================================================

def make_two_panel_heatmap(baseline_mat, real_mat, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.8,
    })

    vmin = min(baseline_mat.min(), real_mat.min())
    vmax = max(baseline_mat.max(), real_mat.max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)

    im0 = axes[0].imshow(baseline_mat, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("Shuffled baseline")
    axes[0].set_xlabel("Window index")
    axes[0].set_ylabel("Window index")

    im1 = axes[1].imshow(real_mat, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("PG-19")
    axes[1].set_xlabel("Window index")
    axes[1].set_ylabel("Window index")

    for ax in axes:
        ax.tick_params(length=7, width=1.8)

    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label("JS divergence")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


#%%
# ============================================================
# Step 4: Load PG-19 books
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

print("Number of training books:", len(train_books_encoded))
print("First 5 train book lengths:", [len(x) for x in train_books_encoded[:5]])

#%%
# ============================================================
# Step 5: Choose one training book and analysis region
# ============================================================

BOOK_INDEX = 0          # change if you want another book
WINDOW_SIZE = 1024
STRIDE = 1024
NUM_WINDOWS = 200
SEED = 0

book_encoded = train_books_encoded[BOOK_INDEX]
analysis_tokens = WINDOW_SIZE + (NUM_WINDOWS - 1) * STRIDE

if len(book_encoded) < analysis_tokens:
    raise ValueError(
        f"Chosen book length {len(book_encoded):,} is shorter than "
        f"required analysis length {analysis_tokens:,}."
    )

analysis_data = book_encoded[:analysis_tokens]
vocab_size = len(stoi)

print(f"Using book index: {BOOK_INDEX}")
print(f"Book length: {len(book_encoded):,}")
print(f"Using {len(analysis_data):,} tokens")
print(f"Window size: {WINDOW_SIZE}")
print(f"Stride: {STRIDE}")
print(f"Num windows: {NUM_WINDOWS}")

#%%
# ============================================================
# Step 6: Real PG-19 pairwise JS matrix
# ============================================================

real_histograms = compute_window_histograms(
    token_ids=analysis_data,
    vocab_size=vocab_size,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    num_windows=NUM_WINDOWS,
)

real_pairwise_js = compute_pairwise_js_matrix(real_histograms)

#%%
# ============================================================
# Step 7: Shuffled baseline pairwise JS matrix
# ============================================================

rng = np.random.default_rng(SEED)
shuffled_data = analysis_data.copy()
rng.shuffle(shuffled_data)

baseline_histograms = compute_window_histograms(
    token_ids=shuffled_data,
    vocab_size=vocab_size,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    num_windows=NUM_WINDOWS,
)

baseline_pairwise_js = compute_pairwise_js_matrix(baseline_histograms)

#%%
# ============================================================
# Step 8: Save figure
# ============================================================

save_dir = "../plots"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "pg19_nonstationarity.pdf")

make_two_panel_heatmap(
    baseline_mat=baseline_pairwise_js,
    real_mat=real_pairwise_js,
    save_path=save_path,
)

print(f"Saved figure to: {os.path.abspath(save_path)}")
# %%
