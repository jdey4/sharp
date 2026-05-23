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
import argparse
import random
from joblib import Parallel, delayed


# ============================================================
# Device
# ============================================================
device = "cpu"
print("Using device:", device, flush=True)


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
def build_model(device, recon_threshold):
    model = Model(
        total_layers=5,
        head_type="film",
        memory_type="multihead",
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
        recon_threshold=recon_threshold,

        bad_init=False,
        device=device,
    )

    return model


# ============================================================
# Evaluation helpers
# ============================================================
@torch.no_grad()
def evaluate_dataset_existing_model(eval_model, eval_dataset, device, max_eval_tokens=None):
    """
    Evaluate one split with an already-loaded eval model.

    The model is reset inside this function so forward/current/backward
    splits are evaluated independently.
    """
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

    return avg_bpc, avg_acc, total_count


@torch.no_grad()
def evaluate_threeway(
    train_model,
    backward_dataset,
    current_dataset,
    forward_dataset,
    device,
    recon_threshold,
    max_eval_tokens=None,
):
    """
    Load train model once, then evaluate:
      - backward: earliest stream region
      - current: recent stream region
      - forward: held-out future region
    """
    eval_model = build_model(
        device=device,
        recon_threshold=recon_threshold,
    )

    eval_model.load_state_dict(copy.deepcopy(train_model.state_dict()))
    eval_model.eval()

    backward_bpc, backward_acc, backward_count = evaluate_dataset_existing_model(
        eval_model=eval_model,
        eval_dataset=backward_dataset,
        device=device,
        max_eval_tokens=max_eval_tokens,
    )

    current_bpc, current_acc, current_count = evaluate_dataset_existing_model(
        eval_model=eval_model,
        eval_dataset=current_dataset,
        device=device,
        max_eval_tokens=max_eval_tokens,
    )

    forward_bpc, forward_acc, forward_count = evaluate_dataset_existing_model(
        eval_model=eval_model,
        eval_dataset=forward_dataset,
        device=device,
        max_eval_tokens=max_eval_tokens,
    )

    del eval_model

    return {
        "forward_bpc": forward_bpc,
        "forward_acc": forward_acc,
        "forward_eval_count": forward_count,

        "current_bpc": current_bpc,
        "current_acc": current_acc,
        "current_eval_count": current_count,

        "backward_bpc": backward_bpc,
        "backward_acc": backward_acc,
        "backward_eval_count": backward_count,
    }


def make_eval_datasets(encoded, train_tokens, samples_seen, eval_tokens, short_term_memory):
    """
    backward: earliest eval_tokens from the stream.
    current: most recent eval_tokens seen so far.
    forward: held-out eval_tokens after train_tokens.
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
# Helpers
# ============================================================
def threshold_to_tag(threshold):
    """
    Examples:
      0      -> tau_0
      1e-3   -> tau_1em03
      1e-2   -> tau_1em02
      1e-1   -> tau_1em01
    """
    if float(threshold) == 0.0:
        return "tau_0"

    tag = f"{threshold:.0e}"
    tag = tag.replace("-", "m").replace("+", "").replace(".", "p")
    return f"tau_{tag}"


def parse_thresholds(thresholds_str):
    vals = []
    for item in thresholds_str.split(","):
        item = item.strip()
        if len(item) == 0:
            continue
        vals.append(float(item))
    return vals


# ============================================================
# One threshold run
# ============================================================
def run_threshold_condition(
    tau,
    encoded,
    train_tokens,
    eval_tokens,
    short_term_memory,
    eval_every,
    sleep_every,
    sleep_total_steps,
    max_train_steps,
    max_eval_tokens,
    save_root,
    seed,
    worker_id,
    show_tqdm,
):
    set_seed(seed + worker_id)

    tau = float(tau)
    tau_tag = threshold_to_tag(tau)
    condition_name = f"threshold_{tau_tag}"

    print(
        f"\n[START | worker={worker_id}] tau={tau} | tau_tag={tau_tag}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Output paths
    # KEEPING SAME FILE/DIRECTORY NAMES AS BEFORE
    # ------------------------------------------------------------
    partial_dir = os.path.join(
        save_root,
        "pickle_files",
        "text8_threshold_sweep_partial",
        tau_tag,
    )

    model_dir = os.path.join(
        save_root,
        "saved_models",
        "text8_threshold_sweep",
        tau_tag,
    )

    final_dir = os.path.join(
        save_root,
        "pickle_files",
        "text8_threshold_sweep",
    )

    os.makedirs(partial_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    partial_path = os.path.join(partial_dir, f"{tau_tag}_partial.pkl")
    partial_csv_path = os.path.join(partial_dir, f"{tau_tag}_partial.csv")

    latest_model_path = os.path.join(model_dir, f"{tau_tag}_latest.pt")
    final_model_path = os.path.join(model_dir, f"{tau_tag}_final.pt")

    final_path = os.path.join(final_dir, f"text8_threshold_sweep_{tau_tag}.pkl")
    final_csv_path = os.path.join(final_dir, f"text8_threshold_sweep_{tau_tag}.csv")

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    train_encoded = encoded[:train_tokens]

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

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = build_model(
        device=device,
        recon_threshold=tau,
    )

    print(
        f"[MODEL | worker={worker_id} | {tau_tag}] "
        f"requested_tau={tau} | model.recon_threshold={model.recon_threshold}",
        flush=True,
    )

    model.summary()
    model.reset_model()
    model.train()

    # ------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------
    ii = 0
    h_ = None
    results = []

    correct_ring = np.zeros(1000, dtype=np.float32)
    bpc_ring = np.zeros(1000, dtype=np.float32)

    # Windowed and cumulative wake memory-update counters
    interval_update_count = 0
    interval_total_count = 0

    cumulative_update_count = 0
    cumulative_total_count = 0

    interval_recon_losses = []
    interval_recon_emas = []

    pbar = tqdm(
        train_loader,
        desc=f"Training {tau_tag}",
        position=worker_id,
        leave=True,
        disable=(not show_tqdm),
    )

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits, loss, recon_loss, h_ = model.wake_step(x, y, h_)

        ii += 1

        # --------------------------------------------------------
        # Wake memory-update rate
        #
        # This mirrors the gate inside wake_step:
        #     recon_loss_ema > recon_threshold
        #
        # Sleep is downstream of this gate because a wake memory
        # update sets model.sleeping=True.
        # --------------------------------------------------------
        did_memory_update = float(model.recon_loss_ema > model.recon_threshold)

        interval_update_count += did_memory_update
        interval_total_count += 1

        cumulative_update_count += did_memory_update
        cumulative_total_count += 1

        interval_recon_losses.append(float(recon_loss))
        interval_recon_emas.append(float(model.recon_loss_ema))

        # --------------------------------------------------------
        # Training window diagnostics
        # --------------------------------------------------------
        with torch.no_grad():
            ring_idx = ii % 1000

            bpc_ring[ring_idx] = float(compute_bpc(logits, y))

            pred_tok = logits.argmax(dim=-1)
            correct_ring[ring_idx] = float((pred_tok[0] == y[0]).item())

        # --------------------------------------------------------
        # Sleep phase, same as previous sleep ablation setup
        # --------------------------------------------------------
        if ii % sleep_every == 0:
            model.sleep_step(total_steps=sleep_total_steps)

        # --------------------------------------------------------
        # Three-way evaluation + intermediate saving
        # --------------------------------------------------------
        if ii % eval_every == 0:
            backward_dataset, current_dataset, forward_dataset = make_eval_datasets(
                encoded=encoded,
                train_tokens=train_tokens,
                samples_seen=ii,
                eval_tokens=eval_tokens,
                short_term_memory=short_term_memory,
            )

            eval_results = evaluate_threeway(
                train_model=model,
                backward_dataset=backward_dataset,
                current_dataset=current_dataset,
                forward_dataset=forward_dataset,
                device=device,
                recon_threshold=tau,
                max_eval_tokens=max_eval_tokens,
            )

            update_rate_window = interval_update_count / max(interval_total_count, 1)
            skip_rate_window = 1.0 - update_rate_window

            update_rate_cumulative = cumulative_update_count / max(cumulative_total_count, 1)
            skip_rate_cumulative = 1.0 - update_rate_cumulative

            row = {
                "condition": condition_name,
                "threshold": tau,
                "threshold_tag": tau_tag,
                "model_recon_threshold": float(model.recon_threshold),
                "samples seen": ii,
                "sleep": 1,

                # Three-way BPC/accuracy
                **eval_results,

                # Backward-compatible aliases
                "eval_bpc": eval_results["forward_bpc"],
                "eval_acc": eval_results["forward_acc"],

                # Training diagnostics
                "train_loss": float(loss),
                "recon_loss": float(recon_loss),
                "recon_loss_ema": float(model.recon_loss_ema),
                "train_acc_window": float(np.mean(correct_ring)),
                "train_bpc_window": float(np.mean(bpc_ring)),

                # Wake memory update-rate diagnostics
                "memory_update_rate_window": float(update_rate_window),
                "memory_skip_rate_window": float(skip_rate_window),
                "memory_update_percent_window": float(100.0 * update_rate_window),
                "memory_skip_percent_window": float(100.0 * skip_rate_window),

                "memory_update_rate_cumulative": float(update_rate_cumulative),
                "memory_skip_rate_cumulative": float(skip_rate_cumulative),
                "memory_update_percent_cumulative": float(100.0 * update_rate_cumulative),
                "memory_skip_percent_cumulative": float(100.0 * skip_rate_cumulative),

                "mean_recon_loss_interval": float(np.mean(interval_recon_losses)),
                "mean_recon_loss_ema_interval": float(np.mean(interval_recon_emas)),

                # Metadata
                "eval_tokens": eval_tokens,
                "short_term_memory": short_term_memory,
                "eval_every": eval_every,
                "sleep_every": sleep_every,
                "sleep_total_steps": sleep_total_steps,
                "seed": seed,
                "worker_id": worker_id,
            }

            results.append(row)

            df_partial = (
                pd.DataFrame(results)
                .sort_values("samples seen")
                .reset_index(drop=True)
            )

            # Save intermediate results
            df_partial.to_pickle(partial_path)
            df_partial.to_csv(partial_csv_path, index=False)

            # Save latest checkpoint
            torch.save(model.state_dict(), latest_model_path)

            print(
                f"[CHECKPOINT | worker={worker_id} | {tau_tag}] "
                f"tau={tau} | model_tau={model.recon_threshold} | "
                f"step={ii:,} | "
                f"F={eval_results['forward_bpc']:.4f} | "
                f"C={eval_results['current_bpc']:.4f} | "
                f"B={eval_results['backward_bpc']:.4f} | "
                f"update_win={100.0 * update_rate_window:.2f}% | "
                f"update_cum={100.0 * update_rate_cumulative:.2f}% | "
                f"recon_ema={model.recon_loss_ema:.4e}",
                flush=True,
            )

            # Reset interval counters
            interval_update_count = 0
            interval_total_count = 0
            interval_recon_losses = []
            interval_recon_emas = []

        if max_train_steps is not None and ii >= max_train_steps:
            print(
                f"[STOP | worker={worker_id} | {tau_tag}] "
                f"Reached max_train_steps={max_train_steps}.",
                flush=True,
            )
            break

    # ------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------
    df = (
        pd.DataFrame(results)
        .sort_values("samples seen")
        .reset_index(drop=True)
    )

    with open(final_path, "wb") as f:
        pickle.dump(df, f)

    df.to_csv(final_csv_path, index=False)

    torch.save(model.state_dict(), final_model_path)

    print(f"\n[DONE | worker={worker_id} | {tau_tag}] Saved partial pickle to: {partial_path}", flush=True)
    print(f"[DONE | worker={worker_id} | {tau_tag}] Saved partial CSV to: {partial_csv_path}", flush=True)
    print(f"[DONE | worker={worker_id} | {tau_tag}] Saved latest model to: {latest_model_path}", flush=True)
    print(f"[DONE | worker={worker_id} | {tau_tag}] Saved final pickle to: {final_path}", flush=True)
    print(f"[DONE | worker={worker_id} | {tau_tag}] Saved final CSV to: {final_csv_path}", flush=True)
    print(f"[DONE | worker={worker_id} | {tau_tag}] Saved final model to: {final_model_path}", flush=True)

    print(df.tail(), flush=True)

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0,1e-1,1e-2,1e-3",
        help="Comma-separated reconstruction thresholds. Default excludes 1e-2 because it already exists.",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_tokens", type=int, default=99_000_000)

    parser.add_argument(
        "--eval_tokens",
        type=int,
        default=100_000,
        help="Tokens per split: forward/current/backward. Actual predictions are eval_tokens - short_term_memory.",
    )

    parser.add_argument("--short_term_memory", type=int, default=4)

    parser.add_argument("--eval_every", type=int, default=100_000)

    parser.add_argument("--sleep_every", type=int, default=20_000)

    parser.add_argument("--sleep_total_steps", type=int, default=1025)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Optional early stop for debugging.",
    )

    parser.add_argument(
        "--max_eval_tokens",
        type=int,
        default=None,
        help="Optional cap on evaluation predictions per split.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-2,
        help="Parallel jobs for threshold sweep.",
    )

    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="Torch threads per worker. Use 1 to avoid CPU oversubscription.",
    )

    parser.add_argument(
        "--save_root",
        type=str,
        default="..",
        help="Root for pickle_files/ and saved_models/.",
    )

    parser.add_argument(
        "--text8_path",
        type=str,
        default="dataset/text8.zip",
    )

    parser.add_argument(
        "--show_tqdm",
        type=int,
        default=0,
        help="0 disables tqdm progress bars in parallel runs to avoid mixed terminal output.",
    )

    args = parser.parse_args()

    torch.set_num_threads(args.torch_num_threads)

    thresholds = parse_thresholds(args.thresholds)

    print("Thresholds:", thresholds, flush=True)
    print("n_jobs:", args.n_jobs, flush=True)
    print("torch_num_threads:", args.torch_num_threads, flush=True)
    print("show_tqdm:", bool(args.show_tqdm), flush=True)

    # ------------------------------------------------------------
    # Load data once in parent
    # ------------------------------------------------------------
    text = download_text8(args.text8_path)
    stoi, itos = build_vocab(text)
    encoded = encode(text, stoi)

    print("Encoded length:", len(encoded), flush=True)
    print("Train tokens:", args.train_tokens, flush=True)
    print("Eval tokens per split:", args.eval_tokens, flush=True)

    all_results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(run_threshold_condition)(
            tau=tau,
            encoded=encoded,
            train_tokens=args.train_tokens,
            eval_tokens=args.eval_tokens,
            short_term_memory=args.short_term_memory,
            eval_every=args.eval_every,
            sleep_every=args.sleep_every,
            sleep_total_steps=args.sleep_total_steps,
            max_train_steps=args.max_train_steps,
            max_eval_tokens=args.max_eval_tokens,
            save_root=args.save_root,
            seed=args.seed,
            worker_id=i,
            show_tqdm=bool(args.show_tqdm),
        )
        for i, tau in enumerate(thresholds)
    )

    flat_results = [row for worker_rows in all_results for row in worker_rows]

    df = (
        pd.DataFrame(flat_results)
        .sort_values(["threshold", "samples seen"])
        .reset_index(drop=True)
    )

    combined_dir = os.path.join(args.save_root, "pickle_files", "text8_threshold_sweep")
    os.makedirs(combined_dir, exist_ok=True)

    combined_path = os.path.join(combined_dir, "text8_threshold_sweep_combined.pkl")
    combined_csv_path = os.path.join(combined_dir, "text8_threshold_sweep_combined.csv")

    with open(combined_path, "wb") as f:
        pickle.dump(df, f)

    df.to_csv(combined_csv_path, index=False)

    print("\nSaved combined results to:", combined_path, flush=True)
    print("Saved combined CSV to:", combined_csv_path, flush=True)
    print(df.head(), flush=True)
    print(df.tail(), flush=True)


if __name__ == "__main__":
    main()


# python text8_threshold_sweep.py \
#   --thresholds 0,1e-3,1e-1 \
#   --n_jobs 3 \
#   --torch_num_threads 1 \
#   --show_tqdm 0