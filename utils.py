import numpy as np
import torch
import torch.nn.functional as F
import math


def _get_member(community, n_members, clockwise=True, train=True, train_percent=.66):

    if train:
        choose = int(np.round(n_members*train_percent))
        random_member = community*n_members + np.random.choice(choose)
    else:
        choose = int(np.round(n_members*train_percent))
        random_member = community*n_members + choose + np.random.choice(n_members-choose)
        
    seq = chr(ord('A')+random_member)

    counter = 0
    next_token = random_member
    while counter < n_members-1:

        if clockwise:
            next_token += 1
        else:
            next_token -= 1
  
        if next_token < community*n_members:
            next_token = (community+1)*n_members-1
        elif next_token == (community+1)*n_members:
            next_token = community*n_members

        seq += chr(ord('A')+next_token)
        counter += 1
    
    return seq 

def get_sequence(n_samples, n_community, n_members, train=True, train_percent=0.66, random_state=0, return_direction=False):
    
    """
    Generate data sequence divided into communities.

    Parameters
    ----------
    n_samples : int
        Total number of tokens to sample.
    n_community : int
        Total number of community.
    n_members : int
        Total number of members in each community.
    train : bool, default=true
        whether generate training or testing sequence.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    out : array of shape [n_samples]
        The generated sequence of tokens.
    """

    if random_state is not None:
        np.random.seed(random_state)

    visits = []
    direction = []
    total_community_visit = int(np.ceil(n_samples/n_members))
    
    for ii in range(total_community_visit):
        visits.append(
            np.random.choice(n_community)
        )

        if ii == 0 or ii == 1:
            direction.append(True)
        elif visits[-2] == visits[-1] and visits[-3] == visits[-1]:
            direction.append(False)
        elif visits[-2] != visits[-1] and visits[-3] == visits[-1]:
            direction.append(True)
        elif visits[-2] == visits[-1] and visits[-3] != visits[-1]:
            direction.append(True)
        else:
            direction.append(False)

    out = ''
    for ii, community in enumerate(visits):
        out += _get_member(community, n_members, clockwise=direction[ii], train=train, train_percent=train_percent) + chr(ord('A')+n_community*n_members)

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
