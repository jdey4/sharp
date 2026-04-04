# ============================================================
# Naive RNN ablation for 3 context requirements: 7, 13, 19
# Saves results in ../pickle_files/naive_rnn_bptt4_ablation.pickle
# ============================================================

import sys
sys.path.append('..')

from sharp.utils import get_sequence, DatasetConverter

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")
print("Using device:", device)

# -----------------------------
# Settings (matched to notebook)
# -----------------------------
total_samples = 5_000_000
n_community = 2
n_members = 3
vocab_size = n_community * n_members + 1

# match your SHARP setting
total_layers = 3
hidden_size = 100
embedding_dim = 30
short_term_memory = 4   # BPTT / local context length

# use the 3 contexts shown in the figure
context_depths = [2, 4, 6]
context_length = [7, 13, 19]

reps = 10
save_path = "../pickle_files/naive_rnn_bptt4_ablation.pickle"

# -----------------------------
# Naive RNN model
# -----------------------------
class NaiveRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 30,
        hidden_size: int = 100,
        num_layers: int = 3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True
        )
        self.readout = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        """
        x: [B, T]
        h: [num_layers, B, H] or None
        """
        emb = self.embedding(x)                 # [B, T, E]
        out, h = self.rnn(emb, h)              # out: [B, T, H]
        logits = self.readout(out[:, -1, :])   # predict next token from final state -> [B, V]
        return logits, h


# -----------------------------
# Training loop
# -----------------------------
res = []
repititions = []
context = []
samples_seen = []

criterion = nn.CrossEntropyLoss()

for rep in tqdm(range(reps), desc="Repetitions"):
    for ctx_id, context_depth in enumerate(context_depths):
        print(f"\n[Rep {rep}] Context depth = {context_depth} (context length = {context_length[ctx_id]})")

        model = NaiveRNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=total_layers
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-12
        )

        # same data pipeline as your SHARP ablation
        data = get_sequence(
            total_samples,
            n_community,
            n_members,
            context_depth=context_depth,
            train_percent=1.0
        )
        dataset = DatasetConverter(data, short_term_memory=short_term_memory)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        model.train()
        h = None
        ii = 0
        correct_ring = np.zeros(1000, dtype=np.float32)

        for x, y in loader:
            x = x.to(device).long()   # expected shape [1, T]
            y = y.to(device).long()   # often [1, 1]

            optimizer.zero_grad()

            logits, h = model(x, h)

            # detach hidden state for truncated BPTT=4 behavior
            if h is not None:
                h = h.detach()

            # flatten target safely
            target = y.view(-1)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ii += 1
                pred_tok = logits.argmax(dim=-1)
                correct_ring[ii % 1000] = (pred_tok[0] == target[0]).item()

                if ii % 1000 == 0:
                    acc = np.sum(correct_ring) / (1000 if ii >= 1000 else ii)
                    res.append(acc)
                    samples_seen.append(ii)
                    repititions.append(rep)
                    context.append(ctx_id)

        # cleanup between runs
        del model, optimizer, data, dataset, loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# -----------------------------
# Save in same style as notebook
# -----------------------------
df_rnn = pd.DataFrame({
    "Accuracy": res,
    "repetition": repititions,
    "context required": context,
    "samples seen": samples_seen
})

os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(df_rnn, f)

print(f"\nSaved naive RNN results to: {save_path}")
print(df_rnn.head())