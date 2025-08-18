import numpy as np

from game.gym_env import FatCatsEnv, FatCatsConfig

class FixtureFatCatsEnv(FatCatsEnv):
    """
    Test double that injects a fixed game state (no RNG) so results are deterministic.
    """

    def __init__(self, game_config: FatCatsConfig, fixture: dict, *, seed: int | None = None):
        self._fixture = fixture
        super().__init__(game_config, seed=seed)

    def _initialize_game_state(self) -> None:
        f = self._fixture
        self.treat_deck = np.asarray(f["treat_deck"], dtype=np.int16)
        self.player_hands = [list(map(int, hand)) for hand in f["player_hands"]]
        self.player_scores = np.zeros(self.game_config.number_of_players, dtype=np.int16)
        self.discarded_treats = np.zeros(self.game_config.treat_deck_size, dtype=np.int16)
        self.current_round_bids = np.zeros(self.game_config.number_of_players, dtype=np.int16)
        self.treat_idx = 0
        self.turn = 0


def _nd_discrete(x: int) -> np.ndarray:
    """
    Return a 0-D ndarray action for Discrete spaces (Gym/Gymnasium friendly).
    .item() on the env side will recover the Python int without copies.
    """
    return np.array(x, dtype=np.int64)  # shape=(), not (1,)


def test_play_one_round_unique_highest_wins() -> None:
    """
    Single-bid mode (allow_multi_bid=False).
    Three players, one treat=20.
    Actions are 0-D ndarrays:
      - P0 plays index 1 -> 10
      - P1 plays index 0 -> 15 (wins)
      - P2 plays index 0 -> 12
    """
    print("Playing one round unique highest wins")
    cfg = FatCatsConfig(
        number_of_players=3,
        trick_cards_per_player=3,
        treat_deck_size=1,                 # exactly one round
        treat_card_values=[5, 10, 15, 20],
        trick_card_values=[5, 10, 12, 15],
        allow_multi_bid=False,
    )
    fixture = {
        "treat_deck": [20],
        "player_hands": [
            [5, 10, 12],   # P0
            [15, 5, 10],   # P1
            [12, 5, 5],    # P2
        ],
    }

    env = FixtureFatCatsEnv(cfg, fixture, seed=123)
    env.reset()

    # Drive the round with 0-D ndarray actions
    obs, reward, terminated, truncated, _ = env.step(_nd_discrete(1))  # P0 -> 10
    assert reward == 0.0 and not terminated and not truncated

    obs, reward, terminated, truncated, _ = env.step(_nd_discrete(0))  # P1 -> 15
    assert reward == 0.0 and not terminated and not truncated

    obs, reward, terminated, truncated, _ = env.step(_nd_discrete(0))  # P2 -> 12; resolves

    # Post-resolution checks
    assert env.treat_idx == 1, "Should advance after resolving the round"
    np.testing.assert_array_equal(env.player_scores, np.array([0, 20, 0], dtype=np.int16))
    assert env.discarded_treats[0] == 0
    assert int(env.current_round_bids.sum()) == 0
    assert env.player_hands[0] == [5, 12]   # 10 popped
    assert env.player_hands[1] == [5, 10]   # 15 popped
    assert env.player_hands[2] == [5, 5]    # 12 popped
    assert terminated is True and truncated is False
    print("Finished playing one round unique highest wins")


def test_play_one_round_tie_highest_discards() -> None:
    """
    Single-bid mode, tie for highest -> treat discarded.
    Actions are 0-D ndarrays.
    """
    print("Playing one round tie highest discards")
    cfg = FatCatsConfig(
        number_of_players=3,
        trick_cards_per_player=3,
        treat_deck_size=1,
        treat_card_values=[5, 10, 15, 20],
        trick_card_values=[5, 10, 12, 15],
        allow_multi_bid=False,
    )
    fixture = {
        "treat_deck": [15],
        "player_hands": [
            [10, 5, 5],  # P0 -> 10
            [10, 5, 5],  # P1 -> 10
            [5, 5, 5],   # P2 -> 5
        ],
    }

    env = FixtureFatCatsEnv(cfg, fixture, seed=321)
    env.reset()

    env.step(_nd_discrete(0))  # 10
    env.step(_nd_discrete(0))  # 10
    _, _, terminated, _, _ = env.step(_nd_discrete(0))  # 5 -> resolves

    assert env.treat_idx == 1
    np.testing.assert_array_equal(env.player_scores, np.array([0, 0, 0], dtype=np.int16))
    assert env.discarded_treats[0] == 15
    assert terminated is True
    print("Finished playing one round tie highest discards")


def test_play_one_round_multi_bid_unique_highest() -> None:
    """
    Multi-bid mode (allow_multi_bid=True).
    Bitmask length == trick_cards_per_player.
    P0 bids 10 (mask [0,1,0]), P1 bids 5+10=15 (mask [1,0,1]) -> wins, P2 bids 12 (mask [1,0,0]).
    """
    print("Playing one round multi-bid unique highest")
    cfg = FatCatsConfig(
        number_of_players=3,
        trick_cards_per_player=3,
        treat_deck_size=1,
        treat_card_values=[5, 10, 15, 20],
        trick_card_values=[5, 10, 12, 15],
        allow_multi_bid=True,
    )
    fixture = {
        "treat_deck": [20],
        "player_hands": [
            [5, 10, 12], # P0
            [5, 10, 5], # P1
            [12, 5, 5], # P2
        ],
    }

    env = FixtureFatCatsEnv(cfg, fixture, seed=42)
    env.reset()

    # Masks are 1-D ndarrays
    env.step(np.array([0, 1, 0], dtype=np.int8))  # P0 -> 10
    env.step(np.array([0, 1, 1], dtype=np.int8))  # P1 -> 10+5=15
    _, _, terminated, _, _ = env.step(np.array([1, 0, 0], dtype=np.int8))  # P2 -> 12

    np.testing.assert_array_equal(env.player_scores, np.array([0, 20, 0], dtype=np.int16))
    assert terminated is True
    print("Finished playing one round multi-bid unique highest")

if __name__ == "__main__":
    test_play_one_round_unique_highest_wins()
    test_play_one_round_tie_highest_discards()
    test_play_one_round_multi_bid_unique_highest()
    print("All tests passed!")