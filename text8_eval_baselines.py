#%%
from source.utils import compute_bpc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import os
import zipfile
import urllib.request
import pickle

#%%
device = "cpu"  # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Choose model type here: "rnn", "gru", or "lstm"
model_type = "rnn"

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
            self.X.append(
                encoded_text[ii:ii + short_term_memory]
            )
            self.y.append(
                encoded_text[ii + short_term_memory]
            )

        self.X = tnsr(np.array(self.X)).long()
        self.y = tnsr(np.array(self.y)).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]

#%%
# ------------------------------------------------------------
# Naive recurrent baseline: RNN / GRU / LSTM
# ------------------------------------------------------------
class CharRNNBaseline(nn.Module):
    def __init__(
        self,
        vocab_size=27,
        embedding_dim=100,
        hidden_size=512,
        num_layers=5,
        model_type="rnn",
        device="cpu",
    ):
        super().__init__()

        self.model_type = model_type.lower()
        self.device = torch.device(device)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.model_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError("model_type must be one of: 'rnn', 'gru', 'lstm'")

        self.readout = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        """
        x: (B, T)
        h:
          - RNN/GRU: (num_layers, B, H)
          - LSTM: ((num_layers, B, H), (num_layers, B, H))
        """
        x = x.to(self.device)
        emb = self.embedding(x)               # (B, T, E)
        out, h = self.rnn(emb, h)            # out: (B, T, H)
        logits = self.readout(out[:, -1, :]) # (B, V)
        return logits, h

#%%
@torch.no_grad()
def evaluate_rnn_model(model, dataset, batch_size=1, device="cpu"):
    """
    Evaluate average accuracy and BPC on a dataset.
    Hidden state is passed through the evaluation stream sequentially.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_correct = 0
    total_bpc = 0.0
    total_count = 0
    h = None

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, h = model(x, h)

        # detach hidden state so it does not keep growing
        if h is not None:
            if isinstance(h, tuple):  # LSTM
                h = tuple(v.detach() for v in h)
            else:
                h = h.detach()

        pred_tok = logits.argmax(dim=-1)
        total_correct += (pred_tok == y).sum().item()
        total_bpc += compute_bpc(logits, y) * x.size(0)
        total_count += x.size(0)

    avg_acc = total_correct / max(total_count, 1)
    avg_bpc = total_bpc / max(total_count, 1)
    return avg_acc, avg_bpc

#%%
text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

train_sample = 10_000_000
short_term_memory = 4
vocab_size = 27

test_data_set_forward = Dataset_converter(
    encoded[-1_000_000:],
    short_term_memory=short_term_memory
)

#%%
# ------------------------------------------------------------
# Evaluation config
# ------------------------------------------------------------
total_model = 1

embedding_dim = 100
hidden_size = 512
num_layers = 5

model_dir = "/Users/jd/sleep_experiment/saved_models/baselines"
output_pickle = f"/Users/jd/sleep_experiment/pickle_files/text8_{model_type}_res.pickle"

acc_forward = []
bpc_forward = []
acc_backward = []
bpc_backward = []
acc_current = []
bpc_current = []

for model_no in range(1, total_model + 1):
    print(f"\nEvaluating {model_type.upper()} model {model_no}")

    test_data_set_backward = Dataset_converter(
        encoded[(model_no - 1) * train_sample : (model_no - 1) * train_sample + 1_000_000],
        short_term_memory=short_term_memory
    )

    test_data_set_current = Dataset_converter(
        encoded[model_no * train_sample - 1_000_000 : model_no * train_sample],
        short_term_memory=short_term_memory
    )

    model = CharRNNBaseline(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        model_type=model_type,
        device=device
    ).to(device)

    # Expected naming pattern:
    #   rnn_model1_text8.pt
    #   gru_model1_text8.pt
    #   lstm_model1_text8.pt
    ckpt_path = os.path.join(model_dir, f"{model_type}_model{model_no}_text8.pt")
    print("Loading checkpoint:", ckpt_path)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    avg_acc, avg_bpc = evaluate_rnn_model(model, test_data_set_forward, device=device)
    print("Forward accuracy", avg_acc)
    print("Forward BPC", avg_bpc)
    acc_forward.append(avg_acc)
    bpc_forward.append(avg_bpc)

    avg_acc, avg_bpc = evaluate_rnn_model(model, test_data_set_backward, device=device)
    print("Backward accuracy", avg_acc)
    print("Backward BPC", avg_bpc)
    acc_backward.append(avg_acc)
    bpc_backward.append(avg_bpc)

    avg_acc, avg_bpc = evaluate_rnn_model(model, test_data_set_current, device=device)
    print("Current accuracy", avg_acc)
    print("Current BPC", avg_bpc)
    acc_current.append(avg_acc)
    bpc_current.append(avg_bpc)

print("\n================ FINAL SUMMARY ================")
print("Model type:", model_type.upper())
print("Average forward accuracy", np.mean(acc_forward), "+-", np.std(acc_forward, ddof=1))
print("Average forward BPC", np.mean(bpc_forward), "+-", np.std(bpc_forward, ddof=1))
print("Average backward accuracy", np.mean(acc_backward), "+-", np.std(acc_backward, ddof=1))
print("Average backward BPC", np.mean(bpc_backward), "+-", np.std(bpc_backward, ddof=1))
print("Average current accuracy", np.mean(acc_current), "+-", np.std(acc_current, ddof=1))
print("Average current BPC", np.mean(bpc_current), "+-", np.std(bpc_current, ddof=1))
print("=============================================\n")

summary = (
    acc_forward,
    bpc_forward,
    acc_backward,
    bpc_backward,
    acc_current,
    bpc_current,
)

with open(output_pickle, "wb") as f:
    pickle.dump(summary, f)

print("Saved results to:", output_pickle)

#%%