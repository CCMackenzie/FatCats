import numpy as np

from game.gym_env import FatCatsEnv, FatCatsConfig


class FixtureFatCatsEnv(FatCatsEnv):
    """
    Test double for FatCatsEnv that injects a fixed game state to ensure determinism.
    We override _initialize_game_state so no RNG/shuffle can cause flakiness.
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


def _mask_first_k(k: int, width: int) -> np.ndarray:
    """
    Build a 1-D MultiBinary action mask of fixed width (trick_cards_per_player),
    with ones in the first k positions. Env should slice to current hand length.
    """
    if k < 0 or k > width:
        raise ValueError("k must be within [0, width]")
    arr = np.zeros(width, dtype=np.int8)
    if k:
        arr[:k] = 1
    return arr


def test_full_short_game_multibid_multiple_rounds_and_runout() -> None:
    """
    Scenario (3 players, multi-bid enabled, 3 rounds):
      Treat deck: [20, 15, 10]
      Initial hands (values are trick cards in hand order):
        P0: [10, 5, 5, 5]
        P1: [12, 10, 5, 5]
        P2: [15, 5, 5, 5]

    Round 1 (treat=20):
      - P0 bids 10           -> mask [0,1,0,0] -> unique value 10
      - P1 bids 12+10 = 22   -> mask [1,1,0,0] -> unique highest -> WINS 20
      - P2 bids 15           -> mask [1,0,0,0]
      Expected after R1:
        scores = [0, 20, 0]
        hands:
          P0: [5,5,5]
          P1: [5,5]
          P2: [5,5,5]

    Round 2 (treat=15) -> tie-highest -> discard:
      - P0 bids 5+5 = 10     -> mask [1,1,0,0]
      - P1 bids 5+5 = 10     -> mask [1,1,0,0]
      - P2 bids 5            -> mask [1,0,0,0]
      Expected after R2:
        scores = [0, 20, 0]
        discards = [0, 15, 0]
        hands:
          P0: [5]
          P1: []       (P1 has RUN OUT of cards)
          P2: [5,5]

    Round 3 (treat=10) — verify run-out handling:
      - P0 bids 5            -> mask [1,0,0,0]
      - P1 has no cards      -> mask [0,0,0,0] (must be treated as PASS=0)
      - P2 bids 5+5 = 10     -> mask [1,1,0,0] -> WINS 10
      Final expected:
        scores = [0, 20, 10]
        all hands empty
        discards = [0, 15, 0]
        terminated=True, truncated=False
    """
    cfg = FatCatsConfig(
        number_of_players=3,
        trick_cards_per_player=4,   # fixed width for MultiBinary space
        treat_deck_size=3,          # short full game of 3 rounds
        treat_card_values=[5, 10, 15, 20],
        trick_card_values=[5, 10, 12, 15],
        allow_multi_bid=True,
    )

    fixture = {
        "treat_deck": [20, 15, 10],
        "player_hands": [
            [10, 5, 5, 5],  # P0
            [12, 10, 5, 5], # P1
            [15, 5, 5, 5],  # P2
        ],
    }

    env = FixtureFatCatsEnv(cfg, fixture, seed=7)
    obs, _ = env.reset()

    # --- Round 1 ---
    obs, reward, terminated, truncated, _ = env.step(np.array([1, 0, 0, 0], dtype=np.int8))  # P0 -> 10
    assert reward == 0.0 and not terminated and not truncated

    obs, reward, terminated, truncated, _ = env.step(np.array([1, 1, 0, 0], dtype=np.int8))  # P1 -> 12+10
    assert reward == 0.0 and not terminated and not truncated

    obs, reward, terminated, truncated, _ = env.step(np.array([1, 0, 0, 0], dtype=np.int8))  # P2 -> 15 (resolves)
    assert env.treat_idx == 1
    np.testing.assert_array_equal(env.player_scores, np.array([0, 20, 0], dtype=np.int16))
    assert env.discarded_treats[0] == 0
    assert env.player_hands[0] == [5, 5, 5]
    assert env.player_hands[1] == [5, 5]
    assert env.player_hands[2] == [5, 5, 5]
    assert not terminated and not truncated

    # --- Round 2 (tie-highest → discard) ---
    obs, reward, terminated, truncated, _ = env.step(np.array([1, 1, 0, 0], dtype=np.int8))  # P0 -> 10
    obs, reward, terminated, truncated, _ = env.step(np.array([1, 1, 0, 0], dtype=np.int8))  # P1 -> 10
    obs, reward, terminated, truncated, _ = env.step(np.array([1, 0, 0, 0], dtype=np.int8))  # P2 -> 5 (resolves)

    assert env.treat_idx == 2
    np.testing.assert_array_equal(env.player_scores, np.array([0, 20, 0], dtype=np.int16))
    np.testing.assert_array_equal(env.discarded_treats, np.array([0, 15, 0], dtype=np.int16))
    assert env.player_hands[0] == [5]
    assert env.player_hands[1] == []          # P1 has run out
    assert env.player_hands[2] == [5, 5]
    assert not terminated and not truncated

    # --- Round 3 (verify run-out handling; P1 passes with empty hand) ---
    obs, reward, terminated, truncated, _ = env.step(np.array([1, 0, 0, 0], dtype=np.int8))  # P0 -> 5
    obs, reward, terminated, truncated, _ = env.step(np.array([0, 0, 0, 0], dtype=np.int8))  # P1 -> PASS (no cards)
    obs, reward, terminated, truncated, _ = env.step(np.array([1, 1, 0, 0], dtype=np.int8))  # P2 -> 10 (resolves)

    assert env.treat_idx == 3
    np.testing.assert_array_equal(env.player_scores, np.array([0, 20, 10], dtype=np.int16))
    np.testing.assert_array_equal(env.discarded_treats, np.array([0, 15, 0], dtype=np.int16))
    assert env.player_hands[0] == []
    assert env.player_hands[1] == []
    assert env.player_hands[2] == []
    assert int(env.current_round_bids.sum()) == 0
    assert terminated is True and truncated is False

    assert obs.shape == env.observation_space.shape
    np.testing.assert_array_equal(obs[-cfg.treat_deck_size:], env.discarded_treats)


if __name__ == "__main__":
    test_full_short_game_multibid_multiple_rounds_and_runout()
    print("Full short multi-bid game test passed.")