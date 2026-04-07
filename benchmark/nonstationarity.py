#%%
import os
import re
import zipfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset

sns.set_context("talk")

# ===================== SETTINGS =====================
fontsize = 22
WINDOW_SIZE = 1024
STRIDE = 1024

# PG-19 loading settings
TARGET_TRAIN_CHARS = 100_000_000
MAX_TRAIN_CHARS_PER_BOOK = 2_000_000
MAX_HOLDOUT_BOOKS = 5
MIN_BOOK_CHARS = 20_000
MAX_EVAL_CHARS_PER_BOOK = 1_000_000
BOOK_INDEX = 0

save_dir = "../plots"
save_path = os.path.join(save_dir, "lag_vs_hellinger_text8_pg19.pdf")


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


def encode_text(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)


# ============================================================
# PG-19 utilities
# ============================================================

def normalize_to_27_vocab(text):
    """
    Convert raw text into the same 27-symbol vocabulary as text8:
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


def encode_fixed_vocab(text, stoi):
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


# ============================================================
# Histogram + distance utilities
# ============================================================

def compute_window_histograms(token_ids, vocab_size, window_size, stride):
    token_ids = np.asarray(token_ids)
    n = len(token_ids)

    if n < window_size:
        raise ValueError(f"window_size={window_size} is larger than sequence length={n}")

    starts = list(range(0, n - window_size + 1, stride))

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


def hellinger_distance(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / p.sum()
    q = q / q.sum()

    return (1.0 / np.sqrt(2.0)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))


def compute_pairwise_distance_matrix(histograms):
    n = len(histograms)
    mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        pi = histograms[i]["probs"]
        for j in range(i, n):
            pj = histograms[j]["probs"]
            d = hellinger_distance(pi, pj)
            mat[i, j] = d
            mat[j, i] = d

    return mat


def average_distance_by_lag(distance_matrix):
    n = distance_matrix.shape[0]
    lags = np.arange(n)
    avg_dist = np.zeros(n, dtype=np.float64)

    for k in range(n):
        avg_dist[k] = np.mean(np.diag(distance_matrix, k=k))

    return lags, avg_dist


# ============================================================
# Plot utility
# ============================================================

def make_lag_plot(lags_text8, avg_text8, lags_pg19, avg_pg19, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    ax.plot(
        lags_text8,
        avg_text8,
        linewidth=3,
        color="tab:blue",
        linestyle="-",
        label="text8"
    )

    ax.plot(
        lags_pg19,
        avg_pg19,
        linewidth=3,
        color="tab:orange",
        linestyle="-",
        label="PG-19"
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel("Lag", fontsize=fontsize)
    ax.set_ylabel("Average Hellinger distance", fontsize=fontsize)

    max_lag = int(max(lags_text8.max(), lags_pg19.max()))
    # xticks = np.linspace(0, max_lag, 5, dtype=int)
    ax.set_xticks([0, 500, 1000, 1500, 2000])

    y_min = min(avg_text8.min(), avg_pg19.min())
    y_max = max(avg_text8.max(), avg_pg19.max())
    y_pad = 0.01 * (y_max - y_min + 1e-12)

    ax.set_ylim(max(0, y_min - 3 * y_pad), y_max + 6 * y_pad)

    # yticks = np.linspace(
    #     round(max(0, y_min - 2 * y_pad), 3),
    #     round(y_max + 2 * y_pad, 3),
    #     4
    # )
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])

    ax.tick_params(labelsize=fontsize - 4)
    ax.ticklabel_format(style="plain", axis="x")

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=fontsize - 2
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


# ============================================================
# Main
# ============================================================

print("\n====================")
print("Loading PG-19")
print("====================")

pg19_stoi, pg19_itos = build_fixed_27_vocab()

train_books_raw, holdout_books_raw, total_train_chars = load_pg19_books_by_token_budget(
    target_train_chars=TARGET_TRAIN_CHARS,
    max_train_chars_per_book=MAX_TRAIN_CHARS_PER_BOOK,
    max_holdout_books=MAX_HOLDOUT_BOOKS,
    min_book_chars=MIN_BOOK_CHARS,
    max_eval_chars_per_book=MAX_EVAL_CHARS_PER_BOOK,
)

train_books_encoded = [encode_fixed_vocab(book, pg19_stoi) for book in train_books_raw]

if BOOK_INDEX >= len(train_books_encoded):
    raise ValueError(
        f"BOOK_INDEX={BOOK_INDEX} is out of range. "
        f"Found only {len(train_books_encoded)} training books."
    )

pg19_book = train_books_encoded[BOOK_INDEX]
pg19_vocab_size = len(pg19_stoi)

pg19_num_windows = 1 + (len(pg19_book) - WINDOW_SIZE) // STRIDE
if pg19_num_windows <= 1:
    raise ValueError("Chosen PG-19 book is too short to produce enough windows.")

pg19_analysis_tokens = WINDOW_SIZE + (pg19_num_windows - 1) * STRIDE
pg19_analysis_data = pg19_book[:pg19_analysis_tokens]

print(f"Using PG-19 book index: {BOOK_INDEX}")
print(f"PG-19 book length: {len(pg19_book):,}")
print(f"PG-19 usable tokens: {len(pg19_analysis_data):,}")
print(f"PG-19 number of windows: {pg19_num_windows:,}")

print("\nComputing PG-19 histograms...")
pg19_histograms = compute_window_histograms(
    token_ids=pg19_analysis_data,
    vocab_size=pg19_vocab_size,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
)

print("Computing PG-19 pairwise Hellinger matrix...")
pg19_dist_matrix = compute_pairwise_distance_matrix(pg19_histograms)

print("Computing PG-19 average distance by lag...")
lags_pg19, avg_pg19 = average_distance_by_lag(pg19_dist_matrix)


print("\n====================")
print("Loading text8")
print("====================")

text8_text = download_text8()
text8_stoi, text8_itos = build_vocab(text8_text)
text8_encoded = encode_text(text8_text, text8_stoi)
text8_vocab_size = len(text8_stoi)

text8_max_windows = 1 + (len(text8_encoded) - WINDOW_SIZE) // STRIDE
if text8_max_windows < pg19_num_windows:
    raise ValueError(
        f"text8 only has {text8_max_windows} windows, which is less than "
        f"PG-19's {pg19_num_windows} windows."
    )

text8_num_windows = pg19_num_windows
text8_analysis_tokens = WINDOW_SIZE + (text8_num_windows - 1) * STRIDE
text8_analysis_data = text8_encoded[:text8_analysis_tokens]

print(f"text8 total length: {len(text8_text):,}")
print(f"text8 usable tokens: {len(text8_analysis_data):,}")
print(f"text8 number of windows used: {text8_num_windows:,}")

print("\nComputing text8 histograms...")
text8_histograms = compute_window_histograms(
    token_ids=text8_analysis_data,
    vocab_size=text8_vocab_size,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
)

print("Computing text8 pairwise Hellinger matrix...")
text8_dist_matrix = compute_pairwise_distance_matrix(text8_histograms)

print("Computing text8 average distance by lag...")
lags_text8, avg_text8 = average_distance_by_lag(text8_dist_matrix)


print("\n====================")
print("Saving plot")
print("====================")

os.makedirs(save_dir, exist_ok=True)

make_lag_plot(
    lags_text8=lags_text8,
    avg_text8=avg_text8,
    lags_pg19=lags_pg19,
    avg_pg19=avg_pg19,
    save_path=save_path,
)

print(f"\nSaved figure to: {os.path.abspath(save_path)}")
# %%
