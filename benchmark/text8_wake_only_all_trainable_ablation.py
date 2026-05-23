from sharp.utils import compute_bpc
from sharp.model.model import Model

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import zipfile
import urllib.request
from tqdm import tqdm
import pandas as pd
import pickle
import copy
from collections import deque
import types


# ============================================================
# Device
# ============================================================
device = "cpu"   # same as previous sleep ablation setup
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
# Model setup: same as previous sleep ablation
# ============================================================
def build_model(device):
    model = Model(
        total_layers=5,
        head_type="film",
        memory_type="multihead",
        num_layers_prediction_head=2,

        vocab_size=27,
        hidden_sizes=[128, 128, 128, 128, 128],
        embedding_dim=30,

        lr_layers=1e-4,
        lr_slowdown_factor=0.25,

        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"weight_decay": 1e-12},

        short_term_memory=4,
        context_tag_buffer_size=20,
        recon_threshold=1e-2,

        bad_init=True,
        device=device,
    )

    return model


# ============================================================
# Monkey-patched wake method:
# all memory modules update during wake; NO SLEEP PHASE
# ============================================================
def wake_step_all_trainable_no_sleep(self, x, y, h_=None, return_context=False):
    """
    Wake-only all-trainable ablation.

    During wake:
      - memory layer 0 is trained with token reconstruction;
      - upper memory layers are trained online with reconstruction losses;
      - prediction heads are trained with next-token CE loss.

    Sleep is completely disabled in the experiment loop.
    """

    device = self._get_runtime_device()

    # ------------------------------------------------------------
    # Enter wake mode
    # ------------------------------------------------------------
    if self.wake is False:
        self.step = 0

        for l in range(self.total_layers):
            H = self.hidden_sizes[l]
            self.h_states[l] = torch.zeros(1, H, device=device)

        # Unfreeze all memories and all heads during wake
        for l in range(self.total_layers):
            for p in self.memories[l].parameters():
                p.requires_grad_(True)

        self._unfreeze_heads()
        self.wake = True

        # Create upper-memory wake optimizers once
        if not hasattr(self, "upper_memory_wake_opts"):
            self.upper_memory_wake_opts = {}
            opt_kwargs = self.optimizer_kwargs or {}

            for l in range(1, self.total_layers):
                self.upper_memory_wake_opts[l] = self.optimizer_class(
                    self.memories[l].parameters(),
                    lr=self.lr_layers,
                    **opt_kwargs,
                )

        # Online reconstruction buffers for upper memories
        self.online_memory_buffers = {}

        for l in range(1, self.total_layers):
            H_lower = self.hidden_sizes[l - 1]

            self.online_memory_buffers[l] = deque(
                [
                    torch.zeros(1, 1, H_lower, device=device)
                    for _ in range(self.short_term_memory)
                ],
                maxlen=self.short_term_memory,
            )

    self.step += 1
    t = self.step

    x = x.to(device)
    y = y.view(-1).long().to(device)

    # ============================================================
    # 1. Layer-0 memory reconstruction update
    # ============================================================
    recon_logit, h0, h_ = self.memories[0](x, h_)

    B, T, V = recon_logit.shape

    recon_loss0 = F.cross_entropy(
        recon_logit.reshape(B * T, V),
        x.reshape(B * T),
    )

    self.recon_loss_ema = 0.1 * recon_loss0.item() + 0.9 * self.recon_loss_ema

    if self.recon_loss_ema > self.recon_threshold:
        self.memory_wake_opt.zero_grad(set_to_none=True)
        recon_loss0.backward()
        self.memory_wake_opt.step()

    # ============================================================
    # 2. Bottom-up state updates
    # ============================================================
    # State propagation remains detached for stability.
    # Upper memories are trained separately below using reconstruction.
    with torch.no_grad():
        for l in range(self.total_layers):
            if self.accelerate is None:
                stride = self.short_term_memory ** l
            else:
                stride = self.accelerate ** l

            if t % stride != 0:
                continue

            if l == 0:
                self.h_states[l] = self.memories[l].encode_step_from_token(
                    x[:, -1],
                    self.h_states[l].unsqueeze(0),
                ).squeeze(0)
            else:
                self.h_states[l] = self.memories[l].encode_step_from_vec(
                    self.h_states[l - 1],
                    self.h_states[l],
                )

    # ============================================================
    # 3. Online wake-time reconstruction updates for upper memories
    # ============================================================
    upper_recon_losses = []

    for l in range(1, self.total_layers):
        if self.accelerate is None:
            stride = self.short_term_memory ** l
        else:
            stride = self.accelerate ** l

        # Update upper memory only when its clock ticks
        if t % stride != 0:
            continue

        lower_state = self.h_states[l - 1].detach().unsqueeze(1)  # (1, 1, H_lower)

        self.online_memory_buffers[l].append(lower_state)

        input_window = torch.cat(
            list(self.online_memory_buffers[l]),
            dim=1,
        ).detach()  # (1, T, H_lower)

        recon_upper, _, _ = self.memories[l](input_window, None)

        recon_loss_l = F.mse_loss(recon_upper, input_window)

        self.upper_memory_wake_opts[l].zero_grad(set_to_none=True)
        recon_loss_l.backward()
        self.upper_memory_wake_opts[l].step()

        upper_recon_losses.append(float(recon_loss_l.detach().item()))

    # ============================================================
    # 4. Top-down context construction through prediction heads
    # ============================================================
    context = None

    for l in reversed(range(self.total_layers)):
        if l == 0:
            logits = self.heads[0](self.h_states[0], context=context)
        else:
            context = self.heads[l](self.h_states[l], context=context)

    logits = logits.squeeze(1)

    # ============================================================
    # 5. Prediction-head update
    # ============================================================
    pred_loss = F.cross_entropy(logits, y)

    for opt in self.head_wake_opts:
        opt.zero_grad(set_to_none=True)

    pred_loss.backward()

    for opt in self.head_wake_opts:
        opt.step()

    if len(upper_recon_losses) > 0:
        mean_upper_recon = float(np.mean(upper_recon_losses))
    else:
        mean_upper_recon = 0.0

    total_recon_report = float(recon_loss0.detach().item()) + mean_upper_recon

    if return_context:
        return (
            logits.detach(),
            pred_loss.item(),
            total_recon_report,
            h_.detach(),
            context.detach() if context is not None else None,
        )

    return logits.detach(), pred_loss.item(), total_recon_report, h_.detach()


# ============================================================
# Evaluation helper
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
# Settings: same as previous sleep ablation
# ============================================================
short_term_memory = 4

train_tokens = 99_000_000
eval_tokens = 300_000

eval_every = 100_000

condition_name = "wake_only_all_trainable"

save_path = "../pickle_files/text8_wake_only_all_trainable.pkl"
partial_dir = "../pickle_files/text8_wake_only_all_trainable_partial"
model_dir = "../saved_models/text8_wake_only_all_trainable"

os.makedirs(partial_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# ============================================================
# Load data
# ============================================================
text = download_text8()
stoi, itos = build_vocab(text)
encoded = encode(text, stoi)

train_encoded = encoded[:train_tokens]
eval_encoded = encoded[90_000_000:90_000_000 + eval_tokens]


# ============================================================
# Run one repetition, no sleep
# ============================================================
def run_experiment():
    print(
        "\n==================== Running condition: "
        "wake_only_all_trainable | NO SLEEP ====================",
        flush=True,
    )

    train_dataset = SequenceDataset(
        train_encoded,
        short_term_memory=short_term_memory,
    )

    eval_dataset = SequenceDataset(
        eval_encoded,
        short_term_memory=short_term_memory,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = build_model(device)

    # Monkey-patch the ablation wake method
    model.wake_step_all_trainable_no_sleep = types.MethodType(
        wake_step_all_trainable_no_sleep,
        model,
    )

    model.summary()
    model.reset_model()
    model.train()

    ii = 0
    h_ = None
    results = []

    correct_ring = np.zeros(1000, dtype=np.float32)
    bpc_ring = np.zeros(1000, dtype=np.float32)

    partial_path = os.path.join(partial_dir, "wake_only_all_trainable_partial.pkl")
    partial_csv_path = os.path.join(partial_dir, "wake_only_all_trainable_partial.csv")
    latest_model_path = os.path.join(model_dir, "wake_only_all_trainable_latest.pt")
    final_model_path = os.path.join(model_dir, "wake_only_all_trainable_final.pt")

    pbar = tqdm(
        train_loader,
        desc="Training (wake_only_all_trainable_no_sleep)",
        leave=True,
    )

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits, loss, recon_loss, h_ = model.wake_step_all_trainable_no_sleep(x, y, h_)

        with torch.no_grad():
            ii += 1
            ring_idx = ii % 1000

            bpc_ring[ring_idx] = float(compute_bpc(logits, y))

            pred_tok = logits.argmax(dim=-1)
            correct_ring[ring_idx] = float((pred_tok[0] == y[0]).item())

        # IMPORTANT:
        # No sleep phase at all.
        # No model.sleep_step(...) call exists in this script.

        if ii % eval_every == 0:
            eval_bpc, eval_acc = evaluate_checkpoint(
                train_model=model,
                eval_dataset=eval_dataset,
                device=device,
                max_eval_tokens=None,
            )

            row = {
                "condition": condition_name,
                "sleep": 0,
                "all_trainable_wake": 1,
                "samples seen": ii,
                "eval_bpc": eval_bpc,
                "eval_acc": eval_acc,
                "train_loss": float(loss),
                "recon_loss": float(recon_loss),
                "train_acc_window": float(np.mean(correct_ring)),
                "train_bpc_window": float(np.mean(bpc_ring)),
            }

            results.append(row)

            df_partial = pd.DataFrame(results)

            # Save intermediate results
            df_partial.to_pickle(partial_path)
            df_partial.to_csv(partial_csv_path, index=False)

            # Save latest checkpoint too
            torch.save(model.state_dict(), latest_model_path)

            print(
                f"[{condition_name}] step={ii:,} | "
                f"train loss={float(loss):.6e} | "
                f"recon loss={float(recon_loss):.6e} | "
                f"train acc={row['train_acc_window']:.4f} | "
                f"train bpc={row['train_bpc_window']:.4f} | "
                f"eval acc={eval_acc:.4f} | "
                f"eval bpc={eval_bpc:.4f}",
                flush=True,
            )

    # Final evaluation if not exactly divisible by eval_every
    if ii % eval_every != 0:
        eval_bpc, eval_acc = evaluate_checkpoint(
            train_model=model,
            eval_dataset=eval_dataset,
            device=device,
            max_eval_tokens=None,
        )

        row = {
            "condition": condition_name,
            "sleep": 0,
            "all_trainable_wake": 1,
            "samples seen": ii,
            "eval_bpc": eval_bpc,
            "eval_acc": eval_acc,
            "train_loss": float(loss),
            "recon_loss": float(recon_loss),
            "train_acc_window": float(np.mean(correct_ring)),
            "train_bpc_window": float(np.mean(bpc_ring)),
        }

        results.append(row)

    df = (
        pd.DataFrame(results)
        .sort_values("samples seen")
        .reset_index(drop=True)
    )

    # Save final results
    with open(save_path, "wb") as f:
        pickle.dump(df, f)

    df.to_csv(save_path.replace(".pkl", ".csv"), index=False)

    torch.save(model.state_dict(), final_model_path)

    print("\nSaved partial pickle to:", partial_path, flush=True)
    print("Saved partial CSV to:", partial_csv_path, flush=True)
    print("Saved final results to:", save_path, flush=True)
    print("Saved final model to:", final_model_path, flush=True)
    print(df.head(), flush=True)
    print(df.tail(), flush=True)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    run_experiment()