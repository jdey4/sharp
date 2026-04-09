# ============================================================
# Parallel SHARP ablation (no sleep) using joblib
# Saves results in the exact same format as the original:
# columns = ['reps', 'samples seen', 'context required', 'Accuracy']
# ============================================================

import sys
sys.path.append('..')

import os
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader

from sharp.utils import get_sequence, DatasetConverter
from sharp.model.model import Model


# -----------------------------
# Device
# -----------------------------
device = "cpu"
print("Using device:", device)

# -----------------------------
# Parameters
# -----------------------------
total_samples, n_community, n_members = 5_000_000, 2, 3
total_layers, head_layers, short_term_memory = 3, 3, 4

context_depths = [2, 4, 6]
context_length = [7, 13, 19]
vocab_size = n_community * n_members + 1

reps = 10
save_path = '../pickle_files/ablation_with_acceleration_no_sleep.pickle'


# -----------------------------
# Single worker
# -----------------------------
def run_single_experiment(rep: int, ctx_id: int, context_depth: int):
    model = Model(
        total_layers=total_layers,
        num_layers_prediction_head=head_layers,

        # ---- Layer sizes ----
        vocab_size=vocab_size,
        hidden_sizes=[100, 100, 100],
        embedding_dim=30,

        # ---- Learning rates per layer ----
        lr_layers=1e-4,

        # ---- Optimizer type ----
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={
            "weight_decay": 1e-12
        },

        # ---- Sleep hyperparameters ----
        short_term_memory=short_term_memory,
        context_tag_buffer_size=100,

        # ---- Misc ----
        recon_threshold=1e-3,
        bad_init=True,
        device=device
    )

    data = get_sequence(
        total_samples,
        n_community,
        n_members,
        context_depth=context_depth,
        train_percent=1.0
    )
    dataset = DatasetConverter(data, short_term_memory=short_term_memory)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ii = 0
    h_ = None
    correct_ring = np.zeros(1000, dtype=np.float32)

    local_res = []
    local_reps = []
    local_context = []
    local_samples_seen = []

    model.train()

    for x, y in loader:
        x = x.to(device).long()
        y = y.to(device).long()

        logits, loss, recon_loss, h_ = model.wake_step(x, y, h_)

        with torch.no_grad():
            ii += 1
            pred_tok = logits.argmax(dim=-1)
            correct_ring[ii % 1000] = (pred_tok[0] == y[0, 0]).item()

            if ii % 1000 == 0:
                acc = np.sum(correct_ring) / (1000 if ii >= 1000 else ii)

                local_res.append(acc)
                local_samples_seen.append(ii)
                local_reps.append(rep)
                local_context.append(ctx_id)

    return {
        "reps": local_reps,
        "samples_seen": local_samples_seen,
        "context": local_context,
        "Accuracy": local_res,
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tasks = [
        (rep, ctx_id, context_depth)
        for rep in range(reps)
        for ctx_id, context_depth in enumerate(context_depths)
    ]

    print(f"Running {len(tasks)} experiments in parallel...")

    results = Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=10
    )(
        delayed(run_single_experiment)(rep, ctx_id, context_depth)
        for rep, ctx_id, context_depth in tasks
    )

    # Flatten back into the exact same format
    repititions = []
    samples_seen = []
    context = []
    res = []

    for out in results:
        repititions.extend(out["reps"])
        samples_seen.extend(out["samples_seen"])
        context.extend(out["context"])
        res.extend(out["Accuracy"])

    df = pd.DataFrame()
    df['reps'] = repititions
    df['samples seen'] = samples_seen
    df['context required'] = context
    df['Accuracy'] = res

    with open(save_path, 'wb') as f:
        pickle.dump(df, f)

    print(f"\nSaved results to: {save_path}")
    print(df.head())