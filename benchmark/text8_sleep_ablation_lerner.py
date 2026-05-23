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
        total_layers=5,
        head_type="film",
        num_layers_prediction_head=2,
        vocab_size=27,
        hidden_sizes=[128, 128, 128, 128, 128],
        embedding_dim=30,
        lr_layers=1e-4,
        lr_slowdown_factor=0.1,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"weight_decay": 1e-12},
        short_term_memory=4,
        context_tag_buffer_size=20,
        recon_threshold=1e-2,
        bad_init=not use_sleep,
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

    return avg_bpc, avg_acc, total_count


def make_eval_datasets(encoded, train_tokens, samples_seen, eval_tokens, short_term_memory):
    """
    Creates three evaluation splits.

    backward: earliest eval_tokens from the stream.
    current: most recent eval_tokens observed by the model.
    forward: held-out eval_tokens after the training stream.
    """

    # Earliest 100k tokens
    backward_encoded = encoded[:eval_tokens]

    # Most recent 100k tokens seen so far
    current_end = min(samples_seen, train_tokens)
    current_start = max(0, current_end - eval_tokens)
    current_encoded = encoded[current_start:current_end]

    # Held-out future 100k tokens
    forward_start = train_tokens
    forward_end = train_tokens + eval_tokens
    forward_encoded = encoded[forward_start:forward_end]

    backward_dataset = SequenceDataset(
        backward_encoded,
        short_term_memory=short_term_memory,
    )

    current_dataset = SequenceDataset(
        current_encoded,
        short_term_memory=short_term_memory,
    )

    forward_dataset = SequenceDataset(
        forward_encoded,
        short_term_memory=short_term_memory,
    )

    return backward_dataset, current_dataset, forward_dataset


# ============================================================
# Settings
# ============================================================
short_term_memory = 4
train_tokens = 99_000_000

# Evaluate 100k tokens for each split.
# Each split gives 100000 - short_term_memory = 99996 next-token predictions.
eval_tokens = 100_000

eval_every = 100_000
sleep_every = 20_000
sleep_total_steps = 1025

save_path = "../pickle_files/text8_sleep_ablation_5M_eval_every_100k_threeway.pickle"
partial_dir = "../pickle_files/text8_sleep_ablation_partial_threeway"
model_dir = "../saved_models/text8_sleep_ablation_parallel_threeway"

os.makedirs(partial_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# ============================================================
# Data loaded once in parent; each worker gets serialized copy
# ============================================================
text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

train_encoded = encoded[:train_tokens]


def run_condition(use_sleep, worker_id):
    mode = "sleep" if use_sleep else "no_sleep"
    print(f"\n==================== Running mode: {mode} ====================", flush=True)

    train_dataset = SequenceDataset(
        train_encoded,
        short_term_memory=short_term_memory,
    )

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
    partial_csv_path = os.path.join(partial_dir, f"{mode}_partial.csv")
    latest_model_path = os.path.join(model_dir, f"{mode}_latest.pt")
    final_model_path = os.path.join(model_dir, f"{mode}_5M_text8.pt")

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
            backward_dataset, current_dataset, forward_dataset = make_eval_datasets(
                encoded=encoded,
                train_tokens=train_tokens,
                samples_seen=ii,
                eval_tokens=eval_tokens,
                short_term_memory=short_term_memory,
            )

            backward_bpc, backward_acc, backward_count = evaluate_checkpoint(
                train_model=model,
                eval_dataset=backward_dataset,
                device=device,
                max_eval_tokens=None,
            )

            current_bpc, current_acc, current_count = evaluate_checkpoint(
                train_model=model,
                eval_dataset=current_dataset,
                device=device,
                max_eval_tokens=None,
            )

            forward_bpc, forward_acc, forward_count = evaluate_checkpoint(
                train_model=model,
                eval_dataset=forward_dataset,
                device=device,
                max_eval_tokens=None,
            )

            row = {
                "condition": mode,
                "sleep": int(use_sleep),
                "samples seen": ii,

                # Three-way evaluation
                "forward_bpc": forward_bpc,
                "forward_acc": forward_acc,
                "forward_eval_count": forward_count,

                "current_bpc": current_bpc,
                "current_acc": current_acc,
                "current_eval_count": current_count,

                "backward_bpc": backward_bpc,
                "backward_acc": backward_acc,
                "backward_eval_count": backward_count,

                # Backward-compatible aliases
                "eval_bpc": forward_bpc,
                "eval_acc": forward_acc,

                # Training diagnostics
                "train_loss": float(loss),
                "recon_loss": float(recon_loss),
                "train_acc_window": float(np.mean(correct_ring)),
                "train_bpc_window": float(np.mean(bpc_ring)),

                # Eval metadata
                "eval_tokens": eval_tokens,
                "short_term_memory": short_term_memory,
            }

            results.append(row)

            df_partial = (
                pd.DataFrame(results)
                .sort_values("samples seen")
                .reset_index(drop=True)
            )

            df_partial.to_pickle(partial_path)
            df_partial.to_csv(partial_csv_path, index=False)
            torch.save(model.state_dict(), latest_model_path)

            print(
                f"[{mode}] step={ii:,} | "
                f"train loss={float(loss):.6e} | "
                f"recon loss={float(recon_loss):.6e} | "
                f"train acc={row['train_acc_window']:.4f} | "
                f"train bpc={row['train_bpc_window']:.4f} | "
                f"forward bpc={forward_bpc:.4f} | "
                f"current bpc={current_bpc:.4f} | "
                f"backward bpc={backward_bpc:.4f}",
                flush=True,
            )

    if ii % eval_every != 0:
        backward_dataset, current_dataset, forward_dataset = make_eval_datasets(
            encoded=encoded,
            train_tokens=train_tokens,
            samples_seen=ii,
            eval_tokens=eval_tokens,
            short_term_memory=short_term_memory,
        )

        backward_bpc, backward_acc, backward_count = evaluate_checkpoint(
            train_model=model,
            eval_dataset=backward_dataset,
            device=device,
            max_eval_tokens=None,
        )

        current_bpc, current_acc, current_count = evaluate_checkpoint(
            train_model=model,
            eval_dataset=current_dataset,
            device=device,
            max_eval_tokens=None,
        )

        forward_bpc, forward_acc, forward_count = evaluate_checkpoint(
            train_model=model,
            eval_dataset=forward_dataset,
            device=device,
            max_eval_tokens=None,
        )

        row = {
            "condition": mode,
            "sleep": int(use_sleep),
            "samples seen": ii,

            "forward_bpc": forward_bpc,
            "forward_acc": forward_acc,
            "forward_eval_count": forward_count,

            "current_bpc": current_bpc,
            "current_acc": current_acc,
            "current_eval_count": current_count,

            "backward_bpc": backward_bpc,
            "backward_acc": backward_acc,
            "backward_eval_count": backward_count,

            "eval_bpc": forward_bpc,
            "eval_acc": forward_acc,

            "train_loss": float(loss),
            "recon_loss": float(recon_loss),
            "train_acc_window": float(np.mean(correct_ring)),
            "train_bpc_window": float(np.mean(bpc_ring)),

            "eval_tokens": eval_tokens,
            "short_term_memory": short_term_memory,
        }

        results.append(row)

        df_partial = (
            pd.DataFrame(results)
            .sort_values("samples seen")
            .reset_index(drop=True)
        )

        df_partial.to_pickle(partial_path)
        df_partial.to_csv(partial_csv_path, index=False)

    torch.save(model.state_dict(), final_model_path)

    return results


if __name__ == "__main__":
    # Change this list depending on what you want to run.
    # [False] = no_sleep only
    # [True] = sleep only
    # [True, False] = both sleep and no_sleep in parallel
    conditions_to_run = [False, True]

    all_results = Parallel(n_jobs=-2, backend="loky", verbose=10)(
        delayed(run_condition)(use_sleep, worker_id=i)
        for i, use_sleep in enumerate(conditions_to_run)
    )

    flat_results = [row for worker_rows in all_results for row in worker_rows]

    df = (
        pd.DataFrame(flat_results)
        .sort_values(["condition", "samples seen"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(df, f)

    df.to_csv(save_path.replace(".pickle", ".csv"), index=False)

    print("\nSaved results to:", save_path, flush=True)
    print("Saved CSV to:", save_path.replace(".pickle", ".csv"), flush=True)
    print(df.head(), flush=True)
    print(df.tail(), flush=True)