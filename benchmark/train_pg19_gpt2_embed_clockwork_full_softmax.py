# train_pg19_gpt2_embed_clockwork_full_softmax.py
#
# PG-19 + GPT-2 tokenization + frozen GPT-2 embedding front-end
# Clockwork RNN recurrent core
# Full softmax over GPT-2 vocabulary
#
# X: [B, T] GPT-2 token ids -> frozen GPT-2 embeddings [B, T, 768]
# Y: next GPT-2 token id
# Objective: cross-entropy over GPT-2 vocab
# Streaming protocol: batch_size=1, shuffle=False, hidden state carried within book,
# hidden state reset between books, truncated BPTT by detaching hidden after each update.

import os
import math
import pickle
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Basic utilities
# ============================================================

def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_bits_per_token(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits:  [B, V]
    targets: [B]
    returns CE in bits/token.
    """
    loss_nats = F.cross_entropy(logits, targets, reduction="mean")
    return float(loss_nats.item() / math.log(2.0))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# GPT-2 tokenizer + frozen embedding front-end
# ============================================================

class GPT2FrozenEmbedder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gpt2 = AutoModel.from_pretrained(model_name)
        self.embedding = self.gpt2.get_input_embeddings().to(device)
        self.embedding.eval()

        for p in self.embedding.parameters():
            p.requires_grad_(False)

        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = self.embedding.weight.shape[1]

        print("GPT-2 model:", model_name)
        print("GPT-2 vocab size:", self.vocab_size)
        print("GPT-2 embedding dim:", self.embed_dim)

    def tokenize(self, text: str) -> np.ndarray:
        return np.array(self.tokenizer.encode(text), dtype=np.int64)

    @torch.no_grad()
    def ids_to_embeddings(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: [B, T] long token IDs
        returns: [B, T, 768] dense GPT-2 embeddings
        """
        return self.embedding(x_ids.to(self.device))


# ============================================================
# PG-19 loading/tokenization
# ============================================================

def _extract_text_field(example):
    for key in ["text", "book_text", "content", "document", "story"]:
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(f"Could not find text field in keys: {list(example.keys())}")


def load_pg19_books_by_gpt2_token_budget(
    embedder: GPT2FrozenEmbedder,
    target_train_tokens: int = 25_000_000,
    max_train_tokens_per_book: Optional[int] = None,
    max_holdout_books: int = 5,
    min_book_tokens: int = 1024,
    max_eval_tokens_per_book: int = 100_000,
    dataset_name: str = "fla-hub/pg19",
) -> Tuple[List[np.ndarray], List[np.ndarray], int, str]:
    print("Loading PG-19 from Hugging Face datasets...")
    ds = load_dataset(dataset_name)

    train_books_encoded: List[np.ndarray] = []
    total_train_tokens = 0

    for ex in tqdm(ds["train"], desc="Collecting train books"):
        raw = _extract_text_field(ex)
        ids = embedder.tokenize(raw)

        if len(ids) < min_book_tokens:
            continue

        if max_train_tokens_per_book is not None:
            ids = ids[:max_train_tokens_per_book]

        if len(ids) < min_book_tokens:
            continue

        remaining = target_train_tokens - total_train_tokens
        if remaining <= 0:
            break

        if len(ids) > remaining:
            ids = ids[:remaining]

        train_books_encoded.append(ids)
        total_train_tokens += len(ids)

        if len(train_books_encoded) % 10 == 0:
            print(
                f"Collected {len(train_books_encoded)} books | "
                f"total GPT-2 tokens = {total_train_tokens:,}",
                flush=True,
            )

        if total_train_tokens >= target_train_tokens:
            break

    holdout_split = "validation" if "validation" in ds else "test"
    holdout_books_encoded: List[np.ndarray] = []

    for ex in tqdm(ds[holdout_split], desc=f"Collecting {holdout_split} books"):
        raw = _extract_text_field(ex)
        ids = embedder.tokenize(raw)

        if len(ids) < min_book_tokens:
            continue

        ids = ids[:max_eval_tokens_per_book]
        holdout_books_encoded.append(ids)

        if len(holdout_books_encoded) >= max_holdout_books:
            break

    print("\nFinal training book count:", len(train_books_encoded))
    print("Total GPT-2 training tokens:", f"{total_train_tokens:,}")
    print("Holdout books:", len(holdout_books_encoded), f"from split='{holdout_split}'")

    return train_books_encoded, holdout_books_encoded, total_train_tokens, holdout_split


# ============================================================
# Dataset
# ============================================================

class PG19GPT2Dataset(Dataset):
    """
    Memory-efficient next-token dataset.
    Stores token IDs and slices windows on demand.
    """
    def __init__(self, token_ids: np.ndarray, short_term_memory: int = 4):
        self.token_ids = token_ids
        self.short_term_memory = short_term_memory
        self.n = len(token_ids) - short_term_memory

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, index):
        x_ids = self.token_ids[index:index + self.short_term_memory]
        y_id = self.token_ids[index + self.short_term_memory]
        return (
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(y_id, dtype=torch.long),
        )


# ============================================================
# GPT-2 embedding Clockwork RNN with full softmax
# ============================================================

class GPT2ClockworkRNNFullSoftmax(nn.Module):
    """
    Clockwork RNN receiving dense GPT-2 embeddings.

    Module i updates only when global_t % period_i == 0.
    Modules are ordered fast -> slow.
    Module i receives recurrent input from modules [0, ..., i].
    Output is a full softmax over GPT-2 vocabulary.
    """
    def __init__(
        self,
        input_dim: int = 768,
        module_hidden_size: int = 256,
        periods: Tuple[int, ...] = (1, 2, 4, 8),
        vocab_size: int = 50257,
        head_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.module_hidden_size = module_hidden_size
        self.periods = list(periods)
        self.num_modules = len(self.periods)
        self.total_hidden_size = self.num_modules * self.module_hidden_size
        self.vocab_size = vocab_size

        self.in_linears = nn.ModuleList()
        self.rec_linears = nn.ModuleList()

        for i in range(self.num_modules):
            self.in_linears.append(nn.Linear(input_dim, module_hidden_size))
            rec_in_dim = (i + 1) * module_hidden_size
            self.rec_linears.append(nn.Linear(rec_in_dim, module_hidden_size, bias=False))

        head = []
        d = self.total_hidden_size
        for _ in range(max(head_layers - 1, 0)):
            head.extend([
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        head.extend([
            nn.LayerNorm(d),
            nn.Linear(d, vocab_size),
        ])
        self.head = nn.Sequential(*head)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, batch_size: int, device: torch.device):
        return [
            torch.zeros(batch_size, self.module_hidden_size, device=device)
            for _ in range(self.num_modules)
        ]

    @staticmethod
    def detach_hidden(h):
        if h is None:
            return None
        return [hi.detach() for hi in h]

    def forward(self, x_dense: torch.Tensor, h=None, start_t: int = 0):
        """
        x_dense: [B, T, input_dim]
        h: list of [B, module_hidden_size], or None
        start_t: global clock index for first token in the input window

        returns:
            logits: [B, vocab_size]
            h: updated hidden list
            end_t: start_t + T
        """
        B, T, _ = x_dense.shape
        device = x_dense.device

        if h is None:
            h = self.init_hidden(B, device)

        for s in range(T):
            current_t = start_t + s
            x_t = x_dense[:, s, :]

            old_h = h
            new_h = []

            for i in range(self.num_modules):
                if current_t % self.periods[i] == 0:
                    rec_input = torch.cat(old_h[:i + 1], dim=-1)
                    h_i = torch.tanh(
                        self.in_linears[i](x_t) + self.rec_linears[i](rec_input)
                    )
                else:
                    h_i = old_h[i]

                new_h.append(h_i)

            h = new_h

        h_cat = torch.cat(h, dim=-1)
        logits = self.head(h_cat)
        end_t = start_t + T
        return logits, h, end_t


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_books(
    model: GPT2ClockworkRNNFullSoftmax,
    embedder: GPT2FrozenEmbedder,
    books_encoded: List[np.ndarray],
    device: torch.device,
    short_term_memory: int = 4,
    max_tokens_per_book: Optional[int] = None,
    name: str = "eval",
):
    total_bits = 0.0
    total_correct = 0
    total_count = 0

    model.eval()

    for book_idx, encoded_book in enumerate(books_encoded):
        if max_tokens_per_book is not None:
            encoded_book = encoded_book[:max_tokens_per_book]

        if len(encoded_book) <= short_term_memory:
            continue

        ds = PG19GPT2Dataset(encoded_book, short_term_memory=short_term_memory)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        # Reset hidden state and clock between books.
        h = None
        global_t = 0

        for x_ids, y_token in tqdm(
            loader,
            desc=f"Evaluating {name} book {book_idx + 1}/{len(books_encoded)}",
            leave=False,
        ):
            y_token = y_token.view(-1).long().to(device)
            x_dense = embedder.ids_to_embeddings(x_ids).to(device)

            logits, h, global_t = model(x_dense, h, start_t=global_t)
            h = model.detach_hidden(h)

            bits = compute_bits_per_token(logits, y_token)
            pred_tok = logits.argmax(dim=-1)

            total_correct += int((pred_tok[0] == y_token[0]).item())
            total_bits += bits
            total_count += 1

    avg_bits = total_bits / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_bits, avg_acc


# ============================================================
# Train / run
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/mps/cpu")
    parser.add_argument("--gpt2_model", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="fla-hub/pg19")

    parser.add_argument("--target_train_tokens", type=int, default=25_000_000)
    parser.add_argument("--max_train_tokens_per_book", type=int, default=-1)
    parser.add_argument("--max_holdout_books", type=int, default=5)
    parser.add_argument("--min_book_tokens", type=int, default=1024)
    parser.add_argument("--max_eval_tokens_per_book", type=int, default=100_000)

    parser.add_argument("--short_term_memory", type=int, default=4)
    parser.add_argument("--periods", type=str, default="1,2,4,8")
    parser.add_argument("--module_hidden_size", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-12)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--model_no", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="../saved_models/pg19_models")
    parser.add_argument("--summary_dir", type=str, default="../pickle_files")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)

    periods = tuple(int(x.strip()) for x in args.periods.split(",") if x.strip())
    max_train_tokens_per_book = (
        None if args.max_train_tokens_per_book < 0 else args.max_train_tokens_per_book
    )

    ensure_dir(args.save_dir)
    ensure_dir(args.summary_dir)

    period_tag = "-".join(str(p) for p in periods)
    save_model_path = os.path.join(
        args.save_dir,
        f"model{args.model_no}_pg19_gpt2_clockwork_fullsoftmax_"
        f"periods{period_tag}_H{args.module_hidden_size}_T{args.short_term_memory}.pt",
    )
    save_summary_path = os.path.join(
        args.summary_dir,
        f"result_pg19_gpt2_clockwork_fullsoftmax_"
        f"periods{period_tag}_H{args.module_hidden_size}_T{args.short_term_memory}.pickle",
    )

    # GPT-2 tokenizer + frozen embedding.
    embedder = GPT2FrozenEmbedder(args.gpt2_model, device)

    # Data.
    train_books_encoded, holdout_books_encoded, total_train_tokens, holdout_split = (
        load_pg19_books_by_gpt2_token_budget(
            embedder=embedder,
            target_train_tokens=args.target_train_tokens,
            max_train_tokens_per_book=max_train_tokens_per_book,
            max_holdout_books=args.max_holdout_books,
            min_book_tokens=args.min_book_tokens,
            max_eval_tokens_per_book=args.max_eval_tokens_per_book,
            dataset_name=args.dataset_name,
        )
    )

    print("Number of training books:", len(train_books_encoded))
    print("Number of holdout books:", len(holdout_books_encoded))
    print("First 5 train book lengths:", [len(x) for x in train_books_encoded[:5]])

    # Model.
    model = GPT2ClockworkRNNFullSoftmax(
        input_dim=embedder.embed_dim,
        module_hidden_size=args.module_hidden_size,
        periods=periods,
        vocab_size=embedder.vocab_size,
        head_layers=args.head_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("\n===== GPT-2 Embedding Clockwork RNN Summary =====")
    print("Input: dense frozen GPT-2 embedding windows")
    print("Target: next GPT-2 token ID")
    print("Output: full softmax over GPT-2 vocabulary")
    print("Periods:", periods)
    print("Num modules:", len(periods))
    print("Hidden size per module:", args.module_hidden_size)
    print("Total hidden size:", len(periods) * args.module_hidden_size)
    print("Head layers:", args.head_layers)
    print("Short-term memory/BPTT window:", args.short_term_memory)
    print("LR:", args.lr)
    print("Device:", device)
    print("Save path:", save_model_path)
    print("===============================================\n")

    should_load = os.path.exists(save_model_path) and not args.force_train

    if should_load:
        print("Found trained Clockwork RNN model at:", save_model_path)
        print("Skipping training and loading model directly for evaluation.")
        state = torch.load(save_model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

    elif args.eval_only:
        raise FileNotFoundError(
            f"--eval_only was set, but no saved model exists at: {save_model_path}"
        )

    else:
        print("No trained Clockwork RNN model found. Starting training.")

        ii = 0
        tokens_seen = 0
        correct_ring = np.zeros(1000, dtype=np.float32)
        bits_ring = np.zeros(1000, dtype=np.float32)

        model.train()

        for rep in range(1):
            for book_idx, encoded_book in enumerate(train_books_encoded):
                print(
                    f"\n=== Training Clockwork RNN on book "
                    f"{book_idx + 1}/{len(train_books_encoded)} "
                    f"| GPT-2 tokens={len(encoded_book):,} ===",
                    flush=True,
                )

                ds = PG19GPT2Dataset(
                    encoded_book,
                    short_term_memory=args.short_term_memory,
                )
                loader = DataLoader(
                    ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )

                # Reset state and clock between books.
                h = None
                global_t = 0

                for x_ids, y_token in tqdm(loader):
                    y_token = y_token.view(-1).long().to(device)

                    with torch.no_grad():
                        x_dense = embedder.ids_to_embeddings(x_ids).to(device)

                    logits, h, global_t = model(x_dense, h, start_t=global_t)
                    loss = F.cross_entropy(logits, y_token)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    if args.grad_clip is not None and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    optimizer.step()

                    # Truncated streaming credit assignment.
                    h = model.detach_hidden(h)

                    with torch.no_grad():
                        ii += 1
                        tokens_seen += 1

                        ring_idx = ii % 1000
                        bits_ring[ring_idx] = compute_bits_per_token(logits, y_token)
                        pred_tok = logits.argmax(dim=-1)
                        correct_ring[ring_idx] = int((pred_tok[0] == y_token[0]).item())

                        if ii % 1000 == 0:
                            acc = float(np.mean(correct_ring))
                            bits = float(np.mean(bits_ring))
                            print(
                                "Iter", ii,
                                f"loss: {float(loss):.8e}",
                                "Acc:", acc,
                                "Bits/token:", bits,
                                f"| GPT-2 tokens seen: {tokens_seen:,}",
                                flush=True,
                            )

        torch.save(model.state_dict(), save_model_path)
        print("\nSaved Clockwork RNN model to:", save_model_path)

    # Final evaluation.
    num_backward_books = min(5, len(train_books_encoded))
    num_current_books = min(5, len(train_books_encoded))

    backward_books = train_books_encoded[:num_backward_books]
    current_books = train_books_encoded[-num_current_books:]
    forward_books = holdout_books_encoded

    print("\nStarting final evaluation...")

    forward_bits, forward_acc = evaluate_books(
        model,
        embedder,
        forward_books,
        device=device,
        short_term_memory=args.short_term_memory,
        max_tokens_per_book=args.max_eval_tokens_per_book,
        name="forward",
    )

    backward_bits, backward_acc = evaluate_books(
        model,
        embedder,
        backward_books,
        device=device,
        short_term_memory=args.short_term_memory,
        max_tokens_per_book=args.max_eval_tokens_per_book,
        name="backward",
    )

    current_bits, current_acc = evaluate_books(
        model,
        embedder,
        current_books,
        device=device,
        short_term_memory=args.short_term_memory,
        max_tokens_per_book=args.max_eval_tokens_per_book,
        name="current",
    )

    print("\n================ FINAL EVALUATION ================")
    print("Model    | GPT-2 Embedding Clockwork RNN Full Softmax")
    print(f"Forward  | Bits/token: {forward_bits:.6f} | Acc: {forward_acc:.6f}")
    print(f"Backward | Bits/token: {backward_bits:.6f} | Acc: {backward_acc:.6f}")
    print(f"Current  | Bits/token: {current_bits:.6f} | Acc: {current_acc:.6f}")
    print("=================================================\n")

    summary = {
        "model": "GPT2_CLOCKWORK_RNN_FULL_SOFTMAX",
        "forward_bits_per_token": forward_bits,
        "forward_acc": forward_acc,
        "backward_bits_per_token": backward_bits,
        "backward_acc": backward_acc,
        "current_bits_per_token": current_bits,
        "current_acc": current_acc,
        "num_train_books": len(train_books_encoded),
        "num_holdout_books": len(holdout_books_encoded),
        "holdout_split": holdout_split,
        "target_train_tokens": args.target_train_tokens,
        "actual_train_tokens": total_train_tokens,
        "max_train_tokens_per_book": max_train_tokens_per_book,
        "max_eval_tokens_per_book": args.max_eval_tokens_per_book,
        "gpt2_model": args.gpt2_model,
        "gpt2_vocab_size": embedder.vocab_size,
        "gpt2_embedding_dim": embedder.embed_dim,
        "periods": periods,
        "num_modules": len(periods),
        "module_hidden_size": args.module_hidden_size,
        "total_hidden_size": len(periods) * args.module_hidden_size,
        "head_layers": args.head_layers,
        "short_term_memory": args.short_term_memory,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "save_model_path": save_model_path,
    }

    with open(save_summary_path, "wb") as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved evaluation summary to:", save_summary_path)


if __name__ == "__main__":
    main()
