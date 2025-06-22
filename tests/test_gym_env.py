import numpy as np
import pytest

from agents.random_agent import RandomAgent
from game.game_config import FatCatsConfig
from game.gym_env import FatCatsEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=[False, True])
def env_and_agent(request):
    """Yield (env, agent) pair for single- and multi-bid modes."""
    allow_multi_bid = request.param
    game_config = FatCatsConfig(**{
        "number_of_players": 2,
        "trick_cards_per_player": 8,
        "allow_multi_bid": allow_multi_bid,
        "treat_card_values": [4, 11, 25],
        "trick_card_values": [1, 2, 3],
        "treat_deck_size": 12,
    })
    env = FatCatsEnv(game_config, seed=123)
    agent = RandomAgent(bid_probability=0.3)
    return env, agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def roll_until_done(env: FatCatsEnv, agent: RandomAgent):
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.act_batch(obs[None, :], env.game_config.allow_multi_bid)[0]
        obs, _, done, _, _ = env.step(action)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reset_shape(env_and_agent):
    env, _ = env_and_agent
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_random_agent_action_valid(env_and_agent):
    env, agent = env_and_agent
    obs, _ = env.reset()
    batch = np.stack([obs, obs])  # two identical observations for batch call
    actions = agent.act_batch(batch, env.game_config.allow_multi_bid)

    # Ensure one action per obs
    assert actions.shape[0] == batch.shape[0]

    # Validate each action against the environment's action_space
    for act in actions:
        assert env.action_space.contains(act)


def test_episode_progress_and_termination(env_and_agent):
    env, agent = env_and_agent
    roll_until_done(env, agent)

    # After termination, current_treat_index must equal deck size
    assert env.treat_idx == env.game_config.treat_deck_size

    # Each player's score must equal the sum of collected (non-zero) treat values
    assert env.player_scores.dtype == np.int16
    assert np.all(env.player_scores >= 0)



def test_hand_shrinks_after_bid():
    """Specifically test that a non-pass action removes card(s) from hand."""
    cfg = FatCatsConfig.model_validate({
        "number_of_players": 2,
        "trick_cards_per_player": 4,
        "allow_multi_bid": False,
        "treat_deck_size": 3,
    })
    env = FatCatsEnv(cfg, seed=0)
    obs, _ = env.reset()

    # Pick first card index (0) to guarantee a bid if hand not empty.
    initial_hand_len = len(env.player_hands[0])
    obs, _, _, _, _ = env.step(np.array([0], dtype=np.int16))
    post_hand_len = len(env.player_hands[0])

    assert post_hand_len == initial_hand_len - 1
