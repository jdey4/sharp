#%%
# 100M regime: 90M-char training segments; evaluate checkpoints from
# train_text8_transformer_100M.py (`..._text8_100M.pt`). Default is one model
# per config (`--total_models 1`, matching a single text8-long run).
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as tnsr
import numpy as np
import os
import zipfile
import urllib.request
import math
import argparse
import pickle
from tqdm import tqdm
from transformer_model import Transformer, CONFIGS


#%%
def download_text8(path="dataset/text8.zip"):
    url = "http://mattmahoney.net/dc/text8.zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading text8...")
        urllib.request.urlretrieve(url, path)
    with zipfile.ZipFile(path) as zf:
        return zf.read(zf.namelist()[0]).decode("utf-8")


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


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
@torch.no_grad()
def evaluate_transformer(model, dataset, device="cpu", pin=False, max_bpc=4.755):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=pin)
    model.eval()

    total_correct = 0
    total_bpc = 0.0
    total_count = 0

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=pin)
        y = y.to(device, non_blocking=pin)

        logits = model(x)
        logits = logits[:, -1, :]

        bpc = F.cross_entropy(logits, y).item() / math.log(2)
        if bpc > max_bpc:
            bpc = max_bpc

        pred_tok = logits.argmax(dim=-1)
        total_correct += (pred_tok == y).sum().item()
        total_bpc += bpc
        total_count += 1

    avg_acc = total_correct / max(total_count, 1)
    avg_bpc = total_bpc / max(total_count, 1)
    return avg_acc, avg_bpc


#%%
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="10M", choices=list(CONFIGS.keys()))
parser.add_argument("--model_dir", type=str, default="../saved_models/transformer_baselines")
parser.add_argument("--total_models", type=int, default=1)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

device = args.device
pin = device.startswith("cuda")
print("Using device:", device)

text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)
train_sample = 90_000_000
short_term_memory = 4

cfg = CONFIGS[args.model_size]

test_data_set_forward = Dataset_converter(
    encoded[-1_000_000:], short_term_memory=short_term_memory
)

#%%
acc_forward = []
bpc_forward = []
acc_backward = []
bpc_backward = []
acc_current = []
bpc_current = []

for model_no in range(1, args.total_models + 1):
    print(f"\nEvaluating transformer {args.model_size} model {model_no}")

    test_data_set_backward = Dataset_converter(
        encoded[(model_no - 1) * train_sample : (model_no - 1) * train_sample + 1_000_000],
        short_term_memory=short_term_memory,
    )
    test_data_set_current = Dataset_converter(
        encoded[model_no * train_sample - 1_000_000 : model_no * train_sample],
        short_term_memory=short_term_memory,
    )

    model = Transformer(**cfg).to(device)

    ckpt_path = os.path.join(
        args.model_dir,
        f"transformer_{args.model_size}_model{model_no}_text8_100M.pt",
    )
    print("Loading:", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    avg_acc, avg_bpc = evaluate_transformer(model, test_data_set_forward, device=device, pin=pin)
    print("Forward accuracy", avg_acc)
    print("Forward BPC", avg_bpc)
    acc_forward.append(avg_acc)
    bpc_forward.append(avg_bpc)

    avg_acc, avg_bpc = evaluate_transformer(model, test_data_set_backward, device=device, pin=pin)
    print("Backward accuracy", avg_acc)
    print("Backward BPC", avg_bpc)
    acc_backward.append(avg_acc)
    bpc_backward.append(avg_bpc)

    avg_acc, avg_bpc = evaluate_transformer(model, test_data_set_current, device=device, pin=pin)
    print("Current accuracy", avg_acc)
    print("Current BPC", avg_bpc)
    acc_current.append(avg_acc)
    bpc_current.append(avg_bpc)

print("\n================ FINAL SUMMARY ================")
print(f"Model: Transformer {args.model_size} (100M regime)")
print(f"Average forward accuracy  {np.mean(acc_forward):.6f} +- {np.std(acc_forward, ddof=1):.6f}")
print(f"Average forward BPC       {np.mean(bpc_forward):.6f} +- {np.std(bpc_forward, ddof=1):.6f}")
print(f"Average backward accuracy {np.mean(acc_backward):.6f} +- {np.std(acc_backward, ddof=1):.6f}")
print(f"Average backward BPC      {np.mean(bpc_backward):.6f} +- {np.std(bpc_backward, ddof=1):.6f}")
print(f"Average current accuracy  {np.mean(acc_current):.6f} +- {np.std(acc_current, ddof=1):.6f}")
print(f"Average current BPC       {np.mean(bpc_current):.6f} +- {np.std(bpc_current, ddof=1):.6f}")
print("=================================================")

summary = (acc_forward, bpc_forward, acc_backward, bpc_backward, acc_current, bpc_current)

os.makedirs("../pickle_files", exist_ok=True)
output_pickle = f"../pickle_files/text8_transformer_{args.model_size}_res_100M.pickle"
with open(output_pickle, "wb") as f:
    pickle.dump(summary, f)
print(f"Saved to {output_pickle}")
#%%
