from sharp.utils import compute_bpc
from sharp.model.model import Model

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import pickle
from collections import deque

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Device
# ============================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


# ============================================================
# GPT-2 tokenizer + frozen embedding front-end
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2 = AutoModel.from_pretrained("gpt2")

gpt2_embed = gpt2.get_input_embeddings().to(device)
gpt2_embed.eval()

for p in gpt2_embed.parameters():
    p.requires_grad_(False)

GPT2_VOCAB_SIZE = tokenizer.vocab_size
GPT2_EMBED_DIM = gpt2_embed.weight.shape[1]

print("GPT-2 vocab size:", GPT2_VOCAB_SIZE)
print("GPT-2 embedding dim:", GPT2_EMBED_DIM)


@torch.no_grad()
def ids_to_gpt2_embeddings(x_ids):
    """
    x_ids: [B, T] long token IDs
    returns: [B, T, 768] dense GPT-2 vectors
    """
    return gpt2_embed(x_ids.to(device))


# ============================================================
# PG-19 loading/tokenization
# ============================================================

def _extract_text_field(example):
    for key in ["text", "book_text", "content", "document", "story"]:
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(f"Could not find text field in keys: {list(example.keys())}")


def tokenize_gpt2(text):
    return np.array(tokenizer.encode(text), dtype=np.int64)


def load_pg19_books_by_gpt2_token_budget(
    target_train_tokens=100_000_000,
    max_train_tokens_per_book=None,
    max_holdout_books=5,
    min_book_tokens=1024,
    max_eval_tokens_per_book=100_000,
):
    print("Loading PG-19 from Hugging Face datasets...")
    ds = load_dataset("fla-hub/pg19")

    train_books_encoded = []
    total_train_tokens = 0

    for ex in tqdm(ds["train"], desc="Collecting train books"):
        raw = _extract_text_field(ex)
        ids = tokenize_gpt2(raw)

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
    holdout_books_encoded = []

    for ex in tqdm(ds[holdout_split], desc=f"Collecting {holdout_split} books"):
        raw = _extract_text_field(ex)
        ids = tokenize_gpt2(raw)

        if len(ids) < min_book_tokens:
            continue

        ids = ids[:max_eval_tokens_per_book]
        holdout_books_encoded.append(ids)

        if len(holdout_books_encoded) >= max_holdout_books:
            break

    print("\nFinal training book count:", len(train_books_encoded))
    print("Total GPT-2 training tokens:", f"{total_train_tokens:,}")
    print("Holdout books:", len(holdout_books_encoded), f"from split='{holdout_split}'")

    return train_books_encoded, holdout_books_encoded, total_train_tokens


# ============================================================
# Dataset: returns token IDs only
# Dense X is created in the loop.
# ============================================================

class PG19GPT2Dataset(Dataset):
    def __init__(self, token_ids, short_term_memory=4):
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
# Sampling helper for sleep
# ============================================================

@torch.no_grad()
def sample_topk(logits, k=50, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    k = min(k, logits.shape[-1])
    v, ix = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(v, dim=-1)
    choice = torch.multinomial(probs, 1)
    return ix.gather(-1, choice).squeeze(-1)


# ============================================================
# Dense-input SHARP steps
# X dense, Y token IDs
# ============================================================

def wake_step_denseX_tokenY(model, x_dense, y_token, h_=None, return_context=False):
    """
    x_dense: [B, T, 768] dense GPT-2 embedding window
    y_token: [B] next GPT-2 token ID

    Layer-0 memory recon loss: MSE on dense GPT-2 embeddings.
    Prediction loss: CE over GPT-2 vocabulary.
    """

    if model.wake is False:
        model.step = 0

        for l in range(model.total_layers):
            H = model.hidden_sizes[l]
            model.h_states[l] = torch.zeros(1, H, device=model.device)

        model._freeze_memories(start_layer=0)
        model._unfreeze_memory(layer=0)
        model._unfreeze_heads()
        model.wake = True

    model.step += 1
    t = model.step

    x_dense = x_dense.to(model.device)
    y_token = y_token.view(-1).long().to(model.device)

    # --------------------------------------------------------
    # Layer-0 dense reconstruction
    # --------------------------------------------------------
    recon_out, h0, h_ = model.memories[0](x_dense, h_)
    recon_loss = nn.functional.mse_loss(recon_out, x_dense)

    model.recon_loss_ema = 0.1 * recon_loss.item() + 0.9 * model.recon_loss_ema

    if model.recon_loss_ema > model.recon_threshold:
        model.memory_wake_opt.zero_grad(set_to_none=True)
        recon_loss.backward()
        model.memory_wake_opt.step()
        model.sleeping = True
        model.store_tags = True

    # --------------------------------------------------------
    # Bottom-up state updates
    # Keep RNN hidden states 3-D during encode_step_from_vec:
    # h_prev: [1, B, H]
    # x_step: [B, 1, D]
    # Stored model.h_states[l]: [B, H]
    # --------------------------------------------------------
    with torch.no_grad():
        for l in range(model.total_layers):
            if model.accelerate is None:
                stride = model.short_term_memory ** l
            else:
                stride = model.accelerate ** l

            if t % stride != 0:
                continue

            if l == 0:
                h_prev = model.h_states[l].unsqueeze(0)   # [1, B, H]
                x_step = x_dense[:, -1:, :]               # [B, 1, 768]

                h_next = model.memories[l].encode_step_from_vec(
                    x_step,
                    h_prev,
                )
                model.h_states[l] = h_next.squeeze(0)     # [B, H]

            else:
                h_prev = model.h_states[l].unsqueeze(0)       # [1, B, H]
                x_step = model.h_states[l - 1].unsqueeze(1)   # [B, 1, H_lower]

                h_next = model.memories[l].encode_step_from_vec(
                    x_step,
                    h_prev,
                )
                model.h_states[l] = h_next.squeeze(0)         # [B, H]

    # --------------------------------------------------------
    # Top-down context construction through pattern blocks
    # --------------------------------------------------------
    context = None

    for l in reversed(range(model.total_layers)):
        if l == 0:
            if model.store_tags and context is not None:
                model.context_tags.append(
                    (model.h_states[0].detach(), context.detach())
                )
                model.store_tags = False

            logits = model.heads[0](model.h_states[0], context=context)

        else:
            context = model.heads[l](model.h_states[l], context=context)

    logits = logits.squeeze(1)  # [B, vocab]

    # --------------------------------------------------------
    # Prediction loss: full softmax CE over GPT-2 token IDs
    # --------------------------------------------------------
    pred_loss = nn.functional.cross_entropy(logits, y_token)

    for opt in model.head_wake_opts:
        opt.zero_grad(set_to_none=True)

    pred_loss.backward()

    for opt in model.head_wake_opts:
        opt.step()

    if return_context:
        return (
            logits.detach(),
            pred_loss.item(),
            recon_loss.item(),
            h_.detach(),
            context.detach() if context is not None else None,
        )

    return logits.detach(), pred_loss.item(), recon_loss.item(), h_.detach()

@torch.no_grad()
def eval_step_denseX_tokenY(model, x_dense, y_token, h_=None):
    model.eval()

    x_dense = x_dense.to(model.device)
    y_token = y_token.view(-1).long().to(model.device)

    recon_out, h0, h_pass = model.memories[0](x_dense, h_)
    recon_loss = nn.functional.mse_loss(recon_out, x_dense)

    model.step += 1
    t = model.step

    for l in range(model.total_layers):
        if model.accelerate is None:
            stride = model.short_term_memory ** l
        else:
            stride = model.accelerate ** l

        if t % stride != 0:
            continue

        if l == 0:
            model.h_states[l] = model.memories[l].encode_step_from_vec(
                x_dense[:, -1:, :],
                model.h_states[l],
            )
        else:
            model.h_states[l] = model.memories[l].encode_step_from_vec(
                model.h_states[l - 1],
                model.h_states[l],
            )

    context = None
    for l in reversed(range(model.total_layers)):
        if l == 0:
            logits = model.heads[0](model.h_states[0], context=context)
        else:
            context = model.heads[l](model.h_states[l], context=context)

    logits = logits.squeeze(1)
    pred_loss = nn.functional.cross_entropy(logits, y_token)

    return logits, pred_loss.item(), recon_loss.item(), h_pass


@torch.no_grad()
def teacher_step_layer0_dense(model, h0_carry, context=None, topk=50):
    """
    h0_carry: [B, H]
    returns h_next: [B, H]
    """

    if h0_carry.dim() == 3:
        h0_carry = h0_carry.squeeze(0)

    z0 = h0_carry.unsqueeze(1)  # [B, 1, H]

    logits = model.heads[0](z0, context=context).squeeze(1)
    next_token = sample_topk(logits, k=topk)

    next_dense = ids_to_gpt2_embeddings(next_token.view(-1, 1)).to(model.device)

    h_prev = h0_carry.unsqueeze(0)  # [1, B, H]
    h_next = model.memories[0].encode_step_from_vec(next_dense, h_prev)

    return h_next.squeeze(0), next_token


def sleep_step_dense(model, total_steps=100, verbose=False, topk=50):
    """
    Dense-input sleep:
    - head samples GPT-2 token IDs
    - token IDs -> GPT-2 dense embeddings
    - layer 0 advances using dense vectors
    - upper memories reconstruct lower hidden-state windows with MSE

    Convention:
        model.h_states[l] / h_states[l] stored as [B, H]
        RNN input windows are [B, T, H]
        RNN hidden states are [1, B, H]
    """

    if model.wake is True:
        model.wake = False

    if model.sleeping is True:
        model.sleeping = False
    else:
        return

    for target_layer in range(1, model.total_layers):
        model._freeze_memories(start_layer=0)
        model._unfreeze_memory(target_layer)
        model._freeze_heads()

        decoder_loss_ema = 0.0
        opt_kwargs = model.optimizer_kwargs or {}

        sleep_opt = model.optimizer_class(
            model.memories[target_layer].parameters(),
            lr=model.lr_layers,
            **opt_kwargs,
        )

        loss_func = nn.MSELoss()

        H_lower = model.hidden_sizes[target_layer - 1]

        input_buffer = deque(
            [
                torch.zeros(1, 1, H_lower, device=model.device)
                for _ in range(model.short_term_memory)
            ],
            maxlen=model.short_term_memory,
        )

        for jj in range(len(model.context_tags)):
            h_states = {}

            # context_tags stores h0 as [B, H]
            h0 = model.context_tags[jj][0]
            if h0.dim() == 3:
                h0 = h0.squeeze(1)
            h_states[0] = h0.detach()  # [B, H]

            h_ = None

            for layer in range(1, target_layer):
                h_states[layer] = None

            for ii in range(total_steps):

                # ------------------------------------------------
                # Advance layer 0 using teacher-generated token
                # ------------------------------------------------
                h_states[0], _ = teacher_step_layer0_dense(
                    model,
                    h_states[0],  # [B, H]
                    context=model.context_tags[jj][1],
                    topk=topk,
                )
                # teacher returns [B, H]
                if h_states[0].dim() == 3:
                    h_states[0] = h_states[0].squeeze(0)

                # ------------------------------------------------
                # Advance intermediate layers up to target_layer - 1
                # ------------------------------------------------
                for layer in range(1, target_layer):
                    if model.accelerate is None:
                        stride = model.short_term_memory ** layer
                    else:
                        stride = model.accelerate ** layer

                    if ii % stride != 0:
                        continue

                    lower = h_states[layer - 1]

                    # lower must be [B, 1, H_lower]
                    if lower.dim() == 2:
                        lower_in = lower.unsqueeze(1)
                    elif lower.dim() == 3:
                        lower_in = lower
                    else:
                        raise ValueError(f"Unexpected lower state shape: {lower.shape}")

                    prev = h_states[layer]

                    if prev is None:
                        h_prev = None
                    else:
                        # prev stored [B, H], RNN hidden needs [1, B, H]
                        if prev.dim() == 2:
                            h_prev = prev.unsqueeze(0)
                        elif prev.dim() == 3:
                            h_prev = prev
                        else:
                            raise ValueError(f"Unexpected prev state shape: {prev.shape}")

                    h_next = model.memories[layer].encode_step_from_vec(
                        lower_in,
                        h_prev,
                    )

                    # store as [B, H]
                    h_states[layer] = h_next.squeeze(0)

                # ------------------------------------------------
                # Train target layer on windows of lower-layer states
                # ------------------------------------------------
                if ii % (model.short_term_memory ** target_layer) != 0:
                    continue

                lower_state = h_states[target_layer - 1]

                # FORCE lower_state to [B, 1, H_lower]
                if lower_state.dim() == 2:
                    lower_state_3d = lower_state.unsqueeze(1)
                elif lower_state.dim() == 3:
                    lower_state_3d = lower_state
                else:
                    raise ValueError(f"Unexpected lower_state shape: {lower_state.shape}")

                input_buffer.append(lower_state_3d.detach())

                inp = torch.cat(list(input_buffer), dim=1)  # [B, T, H_lower]

                recon_out, _, h_ = model.memories[target_layer](inp, h_)
                h_ = h_.detach()

                recon_loss = loss_func(recon_out, inp)
                decoder_loss_ema = 0.1 * recon_loss.item() + 0.9 * decoder_loss_ema

                if decoder_loss_ema > model.recon_threshold:
                    sleep_opt.zero_grad(set_to_none=True)
                    recon_loss.backward()
                    sleep_opt.step()

            if verbose:
                print(
                    "Layer", target_layer,
                    "Sleep tag", jj,
                    "loss:", float(recon_loss.item()),
                    flush=True,
                )

    model.context_tags.clear()

# ============================================================
# Evaluation
# ============================================================

def reset_eval_state(model):
    model.wake = False
    model.store_tags = False
    model.step = 0
    model.recon_loss_ema = 0.0
    model.sleeping = False

    for l in range(model.total_layers):
        H = model.hidden_sizes[l]
        model.h_states[l] = torch.zeros(1, H, device=model.device)


@torch.no_grad()
def evaluate_books(model, books_encoded, short_term_memory=4, max_tokens_per_book=None):
    total_bits = 0.0
    total_correct = 0
    total_count = 0

    model.eval()

    for encoded_book in books_encoded:
        if max_tokens_per_book is not None:
            encoded_book = encoded_book[:max_tokens_per_book]

        if len(encoded_book) <= short_term_memory:
            continue

        ds = PG19GPT2Dataset(encoded_book, short_term_memory=short_term_memory)

        loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        reset_eval_state(model)
        h_ = None

        for x_ids, y_token in loader:
            y_token = y_token.to(model.device)

            x_dense = ids_to_gpt2_embeddings(x_ids).to(model.device)

            logits, pred_loss, recon_loss, h_ = eval_step_denseX_tokenY(
                model,
                x_dense,
                y_token,
                h_,
            )

            bits = compute_bpc(logits, y_token)
            pred_tok = logits.argmax(dim=-1)

            total_correct += (pred_tok[0] == y_token[0]).item()
            total_bits += bits
            total_count += 1

    avg_bits = total_bits / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)

    return avg_bits, avg_acc


# ============================================================
# Settings
# ============================================================

target_train_tokens = 25_000_000
max_train_tokens_per_book = None
max_holdout_books = 5
min_book_tokens = 1024
max_eval_tokens_per_book = 100_000

total_layers = 4
head_layers = 4
short_term_memory = 4

hidden_size = 512
sharp_embedding_dim = 128

sleep_every = 20_000
sleep_total_steps = 257

save_model_path = "../saved_models/pg19_models/model1_pg19_gpt2_denseX_tokenY_fullsoftmax.pt"
save_summary_path = "../pickle_files/result_pg19_gpt2_denseX_tokenY_fullsoftmax.pickle"

os.makedirs("../saved_models/pg19_models", exist_ok=True)
os.makedirs("../pickle_files", exist_ok=True)


# ============================================================
# Load data
# ============================================================

train_books_encoded, holdout_books_encoded, total_train_tokens = (
    load_pg19_books_by_gpt2_token_budget(
        target_train_tokens=target_train_tokens,
        max_train_tokens_per_book=max_train_tokens_per_book,
        max_holdout_books=max_holdout_books,
        min_book_tokens=min_book_tokens,
        max_eval_tokens_per_book=max_eval_tokens_per_book,
    )
)

print("Number of training books:", len(train_books_encoded))
print("Number of holdout books:", len(holdout_books_encoded))
print("First 5 train book lengths:", [len(x) for x in train_books_encoded[:5]])


# ============================================================
# Build model
# ============================================================

model = Model(
    total_layers=total_layers,
    num_layers_prediction_head=head_layers,
    memory_type="multihead",
    head_type="film",

    # Prediction output vocabulary
    vocab_size=GPT2_VOCAB_SIZE,

    # Important: source model must support dense layer-0 input.
    # This says layer-0 memory input is 768-d GPT-2 embedding.
    input_size=GPT2_EMBED_DIM,

    hidden_sizes=[hidden_size] * total_layers,
    embedding_dim=sharp_embedding_dim,
    pretrained_embedding=True,

    lr_layers=1e-4,
    lr_slowdown_factor=0.5,
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs={"weight_decay": 1e-12},

    short_term_memory=short_term_memory,
    context_tag_buffer_size=50,
    recon_threshold=1e-2,
    device=device,
)

model.summary()

print("\nTraining SHARP on PG-19")
print("X: dense GPT-2 embedding windows")
print("Y: next GPT-2 token ID")
print("Output: full softmax over GPT-2 vocabulary")


# ============================================================
# Training
# ============================================================

model.reset_model()

ii = 0
tokens_seen = 0
h_ = None

correct_ring = np.zeros(1000, dtype=np.float32)
bits_ring = np.zeros(1000, dtype=np.float32)

for rep in range(1):
    for book_idx, encoded_book in enumerate(train_books_encoded):
        print(
            f"\n=== Training on book {book_idx + 1}/{len(train_books_encoded)} "
            f"| GPT-2 tokens={len(encoded_book):,} ===",
            flush=True,
        )

        ds = PG19GPT2Dataset(encoded_book, short_term_memory=short_term_memory)

        loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        h_ = None
        model.reset_model()

        for x_ids, y_token in tqdm(loader):
            y_token = y_token.to(model.device)

            with torch.no_grad():
                x_dense = ids_to_gpt2_embeddings(x_ids).to(model.device)

            logits, loss, recon_loss, h_ = wake_step_denseX_tokenY(
                model,
                x_dense,
                y_token,
                h_,
            )

            with torch.no_grad():
                ii += 1
                tokens_seen += 1

                ring_idx = ii % 1000
                bits_ring[ring_idx] = compute_bpc(logits, y_token)

                pred_tok = logits.argmax(dim=-1)
                correct_ring[ring_idx] = (pred_tok[0] == y_token[0]).item()

                if ii % 1000 == 0:
                    acc = float(np.mean(correct_ring))
                    bits = float(np.mean(bits_ring))

                    print(
                        "Iter", ii,
                        f"prediction loss: {loss:.8e}",
                        f"Memory loss: {recon_loss:.8e}",
                        "Acc:", acc,
                        "Bits/token:", bits,
                        f"| GPT-2 tokens seen: {tokens_seen:,}",
                        flush=True,
                    )

            if ii % sleep_every == 0:
                sleep_step_dense(
                    model,
                    total_steps=sleep_total_steps,
                    verbose=False,
                    topk=50,
                )


# ============================================================
# Save model
# ============================================================

torch.save(model.state_dict(), save_model_path)
print("\nSaved model to:", save_model_path)


# ============================================================
# Final evaluation
# ============================================================

num_backward_books = min(5, len(train_books_encoded))
num_current_books = min(5, len(train_books_encoded))

backward_books = train_books_encoded[:num_backward_books]
current_books = train_books_encoded[-num_current_books:]
forward_books = holdout_books_encoded

forward_bits, forward_acc = evaluate_books(
    model,
    forward_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_tokens_per_book,
)

backward_bits, backward_acc = evaluate_books(
    model,
    backward_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_tokens_per_book,
)

current_bits, current_acc = evaluate_books(
    model,
    current_books,
    short_term_memory=short_term_memory,
    max_tokens_per_book=max_eval_tokens_per_book,
)

print("\n================ FINAL EVALUATION ================")
print(f"Forward  | Bits/token: {forward_bits:.6f} | Acc: {forward_acc:.6f}")
print(f"Backward | Bits/token: {backward_bits:.6f} | Acc: {backward_acc:.6f}")
print(f"Current  | Bits/token: {current_bits:.6f} | Acc: {current_acc:.6f}")
print("=================================================\n")


# ============================================================
# Save summary
# ============================================================

summary = {
    "forward_bits_per_token": forward_bits,
    "forward_acc": forward_acc,
    "backward_bits_per_token": backward_bits,
    "backward_acc": backward_acc,
    "current_bits_per_token": current_bits,
    "current_acc": current_acc,
    "num_train_books": len(train_books_encoded),
    "num_holdout_books": len(holdout_books_encoded),
    "target_train_tokens": target_train_tokens,
    "actual_train_tokens": total_train_tokens,
    "max_train_tokens_per_book": max_train_tokens_per_book,
    "max_eval_tokens_per_book": max_eval_tokens_per_book,
    "gpt2_vocab_size": GPT2_VOCAB_SIZE,
    "gpt2_embedding_dim": GPT2_EMBED_DIM,
    "sharp_embedding_dim": sharp_embedding_dim,
    "hidden_size": hidden_size,
    "total_layers": total_layers,
    "head_layers": head_layers,
    "short_term_memory": short_term_memory,
}

with open(save_summary_path, "wb") as handle:
    pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved evaluation summary to:", save_summary_path)