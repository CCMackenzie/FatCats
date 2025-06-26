# benchmarks/bench_step.py
import time, numpy as np
from game.game_config import FatCatsConfig
from game.gym_env import FatCatsEnv
from agents.random_agent import RandomAgent

game_config = FatCatsConfig.model_validate({
    "number_of_players": 3,
    "trick_cards_per_player": 5,
    "allow_multi_bid": False,
    "treat_deck_size": 9,
})
parallel_env = FatCatsEnv(game_config, seed=0)
agent = RandomAgent()

obs, _ = parallel_env.reset()
N_STEPS = 1_000_000
print(f"Running {N_STEPS:,} steps with {game_config.number_of_players} players...")
start = time.perf_counter_ns()
for _ in range(N_STEPS):
    act = agent.act_batch(obs[None, :], game_config.allow_multi_bid)[0]
    obs, _, done, _, _ = parallel_env.step(act)
    if done:
        obs, _ = parallel_env.reset()
elapsed = (time.perf_counter_ns() - start) / 1e9
print(f"{N_STEPS / elapsed:,.0f} steps/s")

# from gymnasium.vector import SyncVectorEnv

# start = time.perf_counter_ns()
# NUM_ENVS = 6                         # start with # CPU cores
# parallel_env = SyncVectorEnv(
#     [lambda i=i: FatCatsEnv(game_config, seed=123+i)
#      for i in range(NUM_ENVS)]
# )
# obs, _ = parallel_env.reset()
# steps = 0
# while steps < 1_000_000:
#     acts = agent.act_batch(obs, game_config.allow_multi_bid)
#     obs, _, terminated, _, _ = parallel_env.step(acts)
#     steps += NUM_ENVS
# end = time.perf_counter_ns()
# elapsed = (end - start) / 1e9
# print(f"{steps / elapsed:,.0f} steps/s ({NUM_ENVS} parallel envs)")