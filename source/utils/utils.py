import numpy as np
import torch
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset
from torch import from_numpy as tnsr
from torch.utils.data import Dataset, DataLoader
import numpy as np

def _get_member(community, n_members, clockwise=True, train=True, train_percent=.66):
    if train:
        choose = int(np.round(n_members * train_percent))
        random_member = community * n_members + np.random.choice(choose)
    else:
        choose = int(np.round(n_members * train_percent))
        random_member = community * n_members + choose + np.random.choice(n_members - choose)

    seq = chr(ord('A') + random_member)

    counter = 0
    next_token = random_member
    while counter < n_members - 1:

        if clockwise:
            next_token += 1
        else:
            next_token -= 1

        if next_token < community * n_members:
            next_token = (community + 1) * n_members - 1
        elif next_token == (community + 1) * n_members:
            next_token = community * n_members

        seq += chr(ord('A') + next_token)
        counter += 1

    return seq


def _direction_from_history(visits, t, K, n_community, mode="hash_parity"):
    """
    Determine direction for visit index t using the previous K visits.

    visits: list[int] of length >= t+1 (communities chosen so far)
    t: current index into visits
    K: how many past visits determine direction
    mode:
      - "hash_parity": parity of a simple rolling hash over last K visits (harder, scalable)
      - "sum_parity" : parity of sum(last K visits) (simpler)
      - "match"      : exact match to a fixed target pattern (very hard, sparse trigger)
    """
    if t < K:
        return True  # default direction for warmup

    hist = visits[t-K:t]  # length K

    if mode == "sum_parity":
        # True if even, False if odd
        return (sum(hist) % 2) == 0

    if mode == "match":
        # fixed target pattern (deterministic). Change if you want.
        target = [(i * 7 + 3) % n_community for i in range(K)]
        return hist == target

    # mode == "hash_parity" (default)
    # rolling hash -> parity. Uses order, so truly K-step dependency.
    state = 0
    for v in hist:
        state = (state * 1315423911 + (v + 1) * 2654435761) & 0xFFFFFFFF
    return (state & 1) == 0


def get_sequence(
    n_samples,
    n_community,
    n_members,
    train=True,
    train_percent=0.66,
    random_state=0,
    return_direction=False,
    context_depth=3,          # <-- NEW: K past visits control direction
    direction_mode="hash_parity"  # "hash_parity", "sum_parity", or "match"
):
    """
    Generate data sequence divided into communities, with direction determined by
    the previous `context_depth` community visits.

    Returns a character stream over:
      - members: 'A'.. (A + n_community*n_members - 1)
      - separator: token id n_community*n_members (as a char)
    """

    if random_state is not None:
        np.random.seed(random_state)

    visits = []
    direction = []
    total_community_visit = int(np.ceil(n_samples / n_members))

    # choose community visits
    for ii in range(total_community_visit):
        visits.append(np.random.choice(n_community))
        direction.append(
            _direction_from_history(
                visits=visits,
                t=ii,
                K=context_depth,
                n_community=n_community,
                mode=direction_mode
            )
        )

    out = ''
    sep_char = chr(ord('A') + n_community * n_members)

    for ii, community in enumerate(visits):
        out += _get_member(
            community,
            n_members,
            clockwise=direction[ii],
            train=train,
            train_percent=train_percent
        ) + sep_char

    if return_direction:
        return out[:n_samples], direction

    return out[:n_samples]
        

def compute_bpc(logits, targets):
    """
    Computes Bits Per Character (BPC) from model logits and target indices.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        targets: Tensor of shape (batch_size, seq_len), with target character indices

    Returns:
        Scalar float: BPC value
    """
    # Flatten logits and targets to compute cross-entropy
    logits = logits.view(-1, logits.size(-1))  # (B*T, V)
    targets = targets.view(-1)                 # (B*T)
    
    # Compute cross-entropy loss in nats
    loss_nats = F.cross_entropy(logits, targets, reduction='mean')  # average over all positions
    
    # Convert from nats to bits
    bpc = loss_nats.item() / math.log(2)
    return bpc

def evaluate_model(model, test_dataset, device="cpu"):
    model.eval()
    model.reset_model()   # reset hidden state progression
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_tokens = 0
    total_correct = 0
    total_bpc = 0.0

    h_ = None

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits, pred_loss, recon_loss, h_ = model.eval_step_no_train(x, y, h_)

            # BPC = CE (nats) / ln(2)
            bpc = pred_loss / math.log(2)

            pred_tok = logits.argmax(dim=-1)

            total_correct += (pred_tok[0] == y.view(-1)[0]).item()
            total_bpc += bpc
            total_tokens += 1

    avg_acc = total_correct / total_tokens
    avg_bpc = total_bpc / total_tokens

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy: {avg_acc:.6f}")
    print(f"BPC:      {avg_bpc:.6f}")
    print("========================\n")

    return avg_acc, avg_bpc


class PatternedSequenceGenerator:
    def __init__(self, tokens):
        self.tokens = tokens

    def _random_transition_matrix(self, n, seed=None):
        """
        Generate a random n x n transition matrix (rows sum to 1).
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        P = rng.random((n, n))        # random positive entries
        P /= P.sum(axis=1, keepdims=True)  # normalize rows
        return P
    
    # ---------------- Cyclic ----------------
    def cyclic_sequence(self, cycle_length, total_length, phase=0):
        seq = []
        for i in range(total_length):
            idx = (i + phase) % cycle_length
            seq.append(self.tokens[idx % len(self.tokens)])
        return seq

    # ---------------- Hierarchical ----------------
    def hierarchical_sequence(self, outer_cycle, inner_cycle, total_length):
        seq = []
        outer_seq = self.cyclic_sequence(outer_cycle, total_length)
        inner_seq = self.cyclic_sequence(inner_cycle, total_length)

        for i in range(total_length):
            # append both outer and inner tokens into flat list
            seq.append(outer_seq[i % len(outer_seq)])
            seq.append(inner_seq[i % len(inner_seq)])
        return seq

    # ---------------- Markov ----------------
    def markov_sequence(self, start_token, length, seed=None):
        transition_matrix = self._random_transition_matrix(
                    len(self.tokens), 
                    seed=seed
                )
        seq = [start_token]
        token_to_index = {t: i for i, t in enumerate(self.tokens)}

        for _ in range(length - 1):
            current_idx = token_to_index[seq[-1]]
            next_token = np.random.choice(self.tokens, p=transition_matrix[current_idx])
            seq.append(next_token)
        return seq

    # ---------------- Noisy ----------------
    def noisy_sequence(self, base_sequence, noise_prob=0.1):
        seq = []
        for token in base_sequence:
            if random.random() < noise_prob:
                seq.append(random.choice(self.tokens))
            else:
                seq.append(token)
        return seq

    # ---------------- Grammar ----------------
    def grammar_sequence(self, grammar_rules, start_token, length):
        seq = [start_token]
        for _ in range(length - 1):
            current_token = seq[-1]
            if current_token in grammar_rules:
                seq.append(random.choice(grammar_rules[current_token]))
            else:
                seq.append(random.choice(self.tokens))
        return seq

    # ---------------- Generate Many ----------------
    def generate_multiple_sequences(self, num_sequences, generator_type, **kwargs):
        sequences = []
        for _ in range(num_sequences):
            if generator_type == "cyclic":
                sequences.append(self.cyclic_sequence(**kwargs))
            elif generator_type == "hierarchical":
                sequences.append(self.hierarchical_sequence(**kwargs))
            elif generator_type == "markov":
                sequences.append(self.markov_sequence(**kwargs))
            elif generator_type == "noisy":
                sequences.append(self.noisy_sequence(**kwargs))
            elif generator_type == "grammar":
                sequences.append(self.grammar_sequence(**kwargs))
        return sequences


# # ===== Example Usage with 7 tokens =====
# tokens = ["A", "B", "C", "D", "E", "F", "G"]
# generator = PatternedSequenceGenerator(tokens)

# # Cyclic
# cyclic_seq = generator.cyclic_sequence(cycle_length=7, total_length=21)
# print("Cyclic Sequence:", cyclic_seq)

# # Hierarchical (flat list, alternating outer and inner)
# hier_seq = generator.hierarchical_sequence(outer_cycle=7, inner_cycle=3, total_length=10)
# print("Hierarchical Sequence:", hier_seq)

# # Markov
# P = np.array([
#     [0.2,0.2,0.1,0.1,0.1,0.2,0.1],
#     [0.1,0.2,0.2,0.1,0.1,0.2,0.1],
#     [0.1,0.1,0.2,0.2,0.1,0.2,0.1],
#     [0.1,0.1,0.1,0.2,0.2,0.2,0.1],
#     [0.1,0.1,0.1,0.1,0.2,0.3,0.1],
#     [0.1,0.2,0.1,0.1,0.2,0.2,0.1],
#     [0.2,0.1,0.1,0.1,0.1,0.3,0.1]
# ])
# markov_seq = generator.markov_sequence(P, start_token="A", length=20)
# print("Markov Sequence:", markov_seq)

# # Noisy
# noisy_seq = generator.noisy_sequence(cyclic_seq, noise_prob=0.2)
# print("Noisy Sequence:", noisy_seq)

# # Grammar
# grammar_rules = {
#     "A": ["B", "C"],
#     "B": ["D", "E"],
#     "C": ["F", "G"],
#     "D": ["A", "F"],
#     "E": ["C", "G"],
#     "F": ["A", "E"],
#     "G": ["B", "D"]
# }
# grammar_seq = generator.grammar_sequence(grammar_rules, start_token="A", length=25)
# print("Grammar Sequence:", grammar_seq)

# =========================
# Dataset
# =========================

class DatasetConverter(Dataset):
    def __init__(self, data, short_term_memory=3):
        self.X = np.zeros((len(data)-1-short_term_memory, short_term_memory), dtype=np.int64)
        self.y = np.zeros((len(data)-1-short_term_memory, 1), dtype=np.int64)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.X[i, j] = ord(data[i+j]) - 65
            self.y[i] = ord(data[i+j+1]) - 65
        self.X = tnsr(self.X).long()
        self.y = tnsr(self.y).long()

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]