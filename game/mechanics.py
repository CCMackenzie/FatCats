import numpy as np
from typing import Sequence


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


def resolve_bids(bids: np.ndarray) -> int | None:
    """
    Return winning player index or None for tie/no-bid.
    If allow_multi_bid=True, bids already contain sum of a player's cards.
    """
    max_bid = bids.max()
    if max_bid == 0:        # everyone passed
        return None
    winners = np.flatnonzero(bids == max_bid)
    return int(winners[0]) if winners.size == 1 else None
