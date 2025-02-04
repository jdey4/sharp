import numpy as np

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

    if random_state != None:
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
        