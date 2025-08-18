import numpy as np
from typing import Sequence
from numba import njit


def build_treat_deck(
    rng: np.random.Generator,
    treat_values: Sequence[int],
    treat_deck_size: int
) -> np.ndarray:
    """Return an Int16 array of length *size*, sampled with replacement."""
    return rng.choice(treat_values, size=treat_deck_size, replace=True).astype(np.int16)


def deal_trick_hands(
    players: int,
    hand_size: int,
    values: Sequence[int],
    rng: np.random.Generator,
) -> list[list[int]]:
    """Sample hand_size trick cards (with replacement) for each player."""
    return [
        rng.choice(values, size=hand_size, replace=True).astype(np.int16).tolist()
        for _ in range(players)
    ]


@njit(cache=True, fastmath=True)
def resolve_bids(bids: np.ndarray) -> int:
    """
    Return winning player index or -1 for tie/no-bid.
    If allow_multi_bid=True, bids already contain sum of a player's cards.
    """
    max_bid = 0
    winner = -1
    tie = False

    for idx in range(bids.size):
        bid = int(bids[idx])
        if bid > max_bid:
            max_bid = bid
            winner = idx
            tie = False
        elif bid == max_bid and bid > 0:
            tie = True
    
    if max_bid == 0 or tie:
        return -1
    return winner
