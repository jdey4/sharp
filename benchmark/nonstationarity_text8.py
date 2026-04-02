#%%
from sharp.utils import DatasetConverter, compute_bpc, evaluate_model
from sharp.model.model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import itertools
from collections import deque
import os
import zipfile
import urllib.request
from tqdm import tqdm
import pickle
import math
import matplotlib.pyplot as plt

#%%
device = "cpu"  # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

#%%
# Step 1: Download and extract text8
def download_text8(path="dataset/text8.zip"):
    url = "http://mattmahoney.net/dc/text8.zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading text8...")
        urllib.request.urlretrieve(url, path)
    with zipfile.ZipFile(path) as zf:
        data = zf.read(zf.namelist()[0]).decode("utf-8")
    return data

# Step 2: Build character-level vocabulary
def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

# Step 3: Encode text into integer tokens
def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)

class Dataset_converter(Dataset):
    def __init__(self, encoded_text, working_memory=1, short_term_memory=8):
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

#%%
# ============================================================
# JS-divergence utilities
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
# Plot utility: two-panel publication-quality heatmap
# ============================================================
def make_two_panel_heatmap(baseline_mat, real_mat, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
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
    axes[1].set_title("text8")
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
# Load text8 using your pipeline
# ============================================================
text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

print("Loaded text length:", len(text))
print("Vocabulary size:", len(stoi))
print("Vocabulary:", sorted(stoi.keys()))

#%%
# ============================================================
# Parameters
# ============================================================
WINDOW_SIZE = 1024
STRIDE = 1024
NUM_WINDOWS = 200
SEED = 0

analysis_tokens = WINDOW_SIZE + (NUM_WINDOWS - 1) * STRIDE
analysis_data = encoded[:analysis_tokens]
vocab_size = len(stoi)

print(f"Using {len(analysis_data):,} tokens")
print(f"Window size: {WINDOW_SIZE}")
print(f"Stride: {STRIDE}")
print(f"Num windows: {NUM_WINDOWS}")

#%%
# ============================================================
# Real text8 pairwise JS matrix
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
# Shuffled baseline pairwise JS matrix
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
# Save figure
# ============================================================
save_dir = "../plots"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "text8_nonstationarity.pdf")

make_two_panel_heatmap(
    baseline_mat=baseline_pairwise_js,
    real_mat=real_pairwise_js,
    save_path=save_path,
)

print(f"Saved figure to: {os.path.abspath(save_path)}")
# %%
