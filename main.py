import numpy as np
from agents.random_agent import RandomAgent
from game.game_config import FatCatsConfig
from game.gym_env import FatCatsEnv
from tqdm import trange


def run_episodes(game_config: FatCatsConfig, total_episodes: int, seed: int = 42) -> list[list[int]]:
    env = FatCatsEnv(game_config, seed=seed)
    agent = RandomAgent()

    # Track results: shape (episodes, players)
    all_scores: list[list[int]] = []

    progress_bar = trange(total_episodes, desc="Simulating", unit="game")
    for _ in progress_bar:
        observation, _ = env.reset()
        done = False
        while not done:
            actions = agent.act_batch(np.expand_dims(observation, 0), game_config.allow_multi_bid)[0]
            observation, _, done, _, _ = env.step(actions)
        all_scores.append(env.player_scores.tolist())

    return all_scores


def summarise(scores: list[list[int]]):
    matrix = np.array(scores)
    per_player_mean = matrix.mean(axis=0)
    per_player_std = matrix.std(axis=0)

    print("\n=== Scoreboard ===")
    for idx, (mu, sd) in enumerate(zip(per_player_mean, per_player_std)):
        print(f"Player {idx}: {mu:.2f} ± {sd:.2f} (treat value)")


def main() -> None:

    config = FatCatsConfig.from_path("config.json")

    print("Running RandomAgent with configuration:")
    print(config.model_dump())

    episode_scores = run_episodes(config, 10, seed=42)
    summarise(episode_scores)

if __name__ == "__main__":
    main()