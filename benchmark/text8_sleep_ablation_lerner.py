from sharp.utils import compute_bpc
from sharp.model.model import Model

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import zipfile
import urllib.request
from tqdm import tqdm
import pandas as pd
import pickle
import copy
from joblib import Parallel, delayed

# ============================================================
# Device
# ============================================================
device = "cpu"   # parallel only recommended on CPU
print("Using device:", device)

# ============================================================
# text8 download / encoding
# ============================================================
def download_text8(path="dataset/text8.zip"):
    url = "http://mattmahoney.net/dc/text8.zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading text8...", flush=True)
        urllib.request.urlretrieve(url, path)

    with zipfile.ZipFile(path) as zf:
        data = zf.read(zf.namelist()[0]).decode("utf-8")
    return data


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)


# ============================================================
# Dataset
# ============================================================
class SequenceDataset(Dataset):
    def __init__(self, encoded_text, short_term_memory=4):
        self.encoded_text = encoded_text
        self.short_term_memory = short_term_memory
        self.n = len(encoded_text) - short_term_memory

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, index):
        x = self.encoded_text[index:index + self.short_term_memory]
        y = self.encoded_text[index + self.short_term_memory]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


# ============================================================
# Model
# ============================================================
def build_model(device, use_sleep=False):
    model = Model(
        total_layers=2,
        num_layers_prediction_head=2,
        vocab_size=27,
        hidden_sizes=[256, 256],
        embedding_dim=50,
        lr_layers=1e-4,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"weight_decay": 1e-12},
        short_term_memory=4,
        context_tag_buffer_size=1000,
        recon_threshold=5e-2,
        bad_init=True, 
        device=device,
    )
    return model


# ============================================================
# Eval helper
# ============================================================
@torch.no_grad()
def evaluate_checkpoint(train_model, eval_dataset, device, max_eval_tokens=None):
    eval_model = build_model(device)
    eval_model.load_state_dict(copy.deepcopy(train_model.state_dict()))
    eval_model.eval()
    eval_model.reset_model()

    loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    total_bpc = 0.0
    total_correct = 0.0
    total_count = 0
    h_ = None

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, pred_loss, recon_loss, h_ = eval_model.eval_step_no_train(x, y, h_)

        total_bpc += float(compute_bpc(logits, y))
        pred_tok = logits.argmax(dim=-1)
        total_correct += float((pred_tok[0] == y[0]).item())
        total_count += 1

        if max_eval_tokens is not None and total_count >= max_eval_tokens:
            break

    avg_bpc = total_bpc / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    del eval_model
    return avg_bpc, avg_acc


# ============================================================
# Settings
# ============================================================
short_term_memory = 4
train_tokens = 99_000_000
eval_tokens = 300_000
eval_every = 300_000
sleep_every = 20_000
sleep_total_steps = 17

save_path = "../pickle_files/text8_sleep_ablation_5M_eval_every_300k_parallel.pickle"
partial_dir = "../pickle_files/text8_sleep_ablation_partial"
model_dir = "../saved_models/text8_sleep_ablation_parallel"

os.makedirs(partial_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ============================================================
# Data loaded once in parent; each worker gets serialized copy
# ============================================================
text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

train_encoded = encoded[:train_tokens]
eval_encoded = encoded[90_000_000:90_000_000 + eval_tokens]


def run_condition(use_sleep, worker_id):
    mode = "sleep" if use_sleep else "no_sleep"
    print(f"\n==================== Running mode: {mode} ====================", flush=True)

    train_dataset = SequenceDataset(train_encoded, short_term_memory=short_term_memory)
    eval_dataset = SequenceDataset(eval_encoded, short_term_memory=short_term_memory)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = build_model(device, use_sleep=use_sleep)
    model.summary()
    model.reset_model()
    model.train()

    ii = 0
    h_ = None
    results = []

    correct_ring = np.zeros(1000, dtype=np.float32)
    bpc_ring = np.zeros(1000, dtype=np.float32)

    partial_path = os.path.join(partial_dir, f"{mode}_partial.pkl")

    pbar = tqdm(
        train_loader,
        desc=f"Training ({mode})",
        position=worker_id,
        leave=True,
    )

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits, loss, recon_loss, h_ = model.wake_step(x, y, h_)

        with torch.no_grad():
            ii += 1
            ring_idx = ii % 1000
            bpc_ring[ring_idx] = float(compute_bpc(logits, y))
            pred_tok = logits.argmax(dim=-1)
            correct_ring[ring_idx] = float((pred_tok[0] == y[0]).item())

        if use_sleep and ii % sleep_every == 0:
            model.sleep_step(total_steps=sleep_total_steps)

        if ii % eval_every == 0:
            eval_bpc, eval_acc = evaluate_checkpoint(
                train_model=model,
                eval_dataset=eval_dataset,
                device=device,
                max_eval_tokens=None,
            )

            row = {
                "condition": mode,
                "sleep": int(use_sleep),
                "samples seen": ii,
                "eval_bpc": eval_bpc,
                "eval_acc": eval_acc,
                "train_loss": float(loss),
                "recon_loss": float(recon_loss),
                "train_acc_window": float(np.mean(correct_ring)),
                "train_bpc_window": float(np.mean(bpc_ring)),
            }
            results.append(row)

            pd.DataFrame(results).to_pickle(partial_path)

            print(
                f"[{mode}] step={ii:,} | "
                f"train loss={float(loss):.6e} | "
                f"recon loss={float(recon_loss):.6e} | "
                f"train acc={row['train_acc_window']:.4f} | "
                f"train bpc={row['train_bpc_window']:.4f} | "
                f"eval acc={eval_acc:.4f} | "
                f"eval bpc={eval_bpc:.4f}",
                flush=True,
            )

    if ii % eval_every != 0:
        eval_bpc, eval_acc = evaluate_checkpoint(
            train_model=model,
            eval_dataset=eval_dataset,
            device=device,
            max_eval_tokens=None,
        )

        row = {
            "condition": mode,
            "sleep": int(use_sleep),
            "samples seen": ii,
            "eval_bpc": eval_bpc,
            "eval_acc": eval_acc,
            "train_loss": float(loss),
            "recon_loss": float(recon_loss),
            "train_acc_window": float(np.mean(correct_ring)),
            "train_bpc_window": float(np.mean(bpc_ring)),
        }
        results.append(row)
        pd.DataFrame(results).to_pickle(partial_path)

    torch.save(model.state_dict(), os.path.join(model_dir, f"{mode}_5M_text8.pt"))

    return results


if __name__ == "__main__":
    all_results = Parallel(n_jobs=2, backend="loky", verbose=10)(
        delayed(run_condition)(use_sleep, worker_id=i)
        for i, use_sleep in enumerate([False, True])
    )

    flat_results = [row for worker_rows in all_results for row in worker_rows]
    df = pd.DataFrame(flat_results).sort_values(["condition", "samples seen"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(df, f)

    print("\nSaved results to:", save_path, flush=True)
    print(df.head(), flush=True)
    print(df.tail(), flush=True)