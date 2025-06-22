"""
Unit tests for game.game_config.FatCatsConfig

Run:
    pytest -q tests/test_game_config.py
"""
import json
from pathlib import Path

import pytest

from game.game_config import FatCatsConfig


def _write_tmp_json(tmp_path: Path, payload: dict, name: str = "cfg.json") -> Path: # type: ignore
    """Utility to persist *payload* under *tmp_path/name* and return that Path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


# ───────────────────────── positive cases ──────────────────────────
def test_from_dict_valid():
    """A minimal valid dict should build a model and keep defaults."""
    config = FatCatsConfig.model_validate(
        {
            "number_of_players": 3,
            "trick_cards_per_player": 7,
        }
    )

    assert config.number_of_players == 3
    assert config.trick_cards_per_player == 7
    # Defaults kick in
    assert config.allow_multi_bid is False
    assert config.treat_deck_size == 9
    assert config.treat_card_values == [5, 10, 15, 20]


def test_from_path_valid(tmp_path: Path):
    """Config loads & validates from JSON file via FatCatsConfig.from_path()."""
    payload: dict[str, int | bool | list[int]] = {
        "number_of_players": 4,
        "trick_cards_per_player": 8,
        "allow_multi_bid": True,
        "treat_card_values": [4, 11, 25],  # custom deck
        "trick_card_values": [1, 2, 3],    # custom tricks
        "treat_deck_size": 12,
    }
    cfg_path = _write_tmp_json(tmp_path, payload) # type: ignore

    cfg = FatCatsConfig.from_path(cfg_path)

    assert cfg.number_of_players == 4
    assert cfg.allow_multi_bid is True
    assert len(cfg.treat_card_values) == 3
    assert cfg.treat_deck_size == 12


# ───────────────────────── negative cases ──────────────────────────
def test_negative_card_value_raises():
    """Any non-positive card value should trigger a ValidationError."""
    with pytest.raises(ValueError):
        FatCatsConfig.model_validate(
            {
                "players": 2,
                "trick_cards_per_player": 5,
                "treat_values": [5, -10, 15],  # negative value
            }
        )