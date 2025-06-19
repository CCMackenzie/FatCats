"""
Unit tests for fatcats.mechanics

Run:
    pytest -q tests/test_mechanics.py
"""
from __future__ import annotations

import numpy as np
import pytest

from game import mechanics


# ────────────────────────────────────────────────────────────────────────────
# build_treat_deck
# ────────────────────────────────────────────────────────────────────────────
def test_build_treat_deck_length_and_values():
    values = [1, 2, 3]
    rng = np.random.default_rng(123)
    deck = mechanics.build_treat_deck(rng, values, treat_deck_size=10)

    # Correct size
    assert len(deck) == 10
    # Only draws from the supplied population
    assert set(deck.tolist()).issubset(values)
    # dtype is int16 for compactness
    assert deck.dtype == np.int16


def test_build_treat_deck_reproducible():
    values = [7, 8, 9]
    deck1 = mechanics.build_treat_deck(np.random.default_rng(42), values, 15)
    deck2 = mechanics.build_treat_deck(np.random.default_rng(42), values, 15)

    # Same seed → identical deck
    assert np.array_equal(deck1, deck2)


# ────────────────────────────────────────────────────────────────────────────
# deal_trick_hands
# ────────────────────────────────────────────────────────────────────────────
def test_deal_trick_hands_shape_and_values():
    players = 3
    hand_size = 5
    values = [4, 5]
    hands = mechanics.deal_trick_hands(players, hand_size, values, np.random.default_rng(0))

    # One hand per player
    assert len(hands) == players
    # Every hand has the requested size and valid card values
    for hand in hands:
        assert len(hand) == hand_size
        assert set(hand).issubset(values)
        # lists should already be int16 (after .tolist())
        assert all(isinstance(c, (int, np.integer)) for c in hand)


# ────────────────────────────────────────────────────────────────────────────
# resolve_bids
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "bids, expected",
    [
        (np.array([5, 3, 1]), 0),      # unique maximum
        (np.array([1, 4, 4]), None),   # tie on max
        (np.array([0, 0, 0]), None),   # everyone passed
    ],
)
def test_resolve_bids(bids, expected): # type: ignore
    assert mechanics.resolve_bids(bids) == expected # type: ignore