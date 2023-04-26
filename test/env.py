import numpy as np
from argparse import ArgumentParser
from hmacl.envs.m2ale.m2ale_agent_vs_bot import CombatEnv

parser = ArgumentParser()
parser.add_argument("--scenario", type=str, default="5u_vs_5u")
parser.add_argument("--camp", type=str, default="offensive")
parser.add_argument("--reward_type", type=str, default="sparse")
parser.add_argument("--time_limit", type=int, default=200)
parser.add_argument("--map_size", type=list, default=[10, 10])
parser.add_argument("--expand_degree", type=int, default=5)
config = parser.parse_args()

env = CombatEnv(**config.__dict__)
env.reset()
num_agents = 5

def log_stats():
    env.reset()
    print("map_size: ", env._env.map_size)
    stats = env.get_stats()
    print("=" * 50)
    print("Initial stats: ")
    print(stats)
    for episode in range(3):
        env.reset()
        done = False
        step = 0
        while not done:
            actions = []
            for a_i in range(num_agents):
                aa1 = np.random.choice(np.nonzero(env.get_avail_agent_actions(a_i))[0])
                actions.append(aa1)
            r, done, i = env.step(actions)
            # print("step: ", step, r, done, i, actions)
            step += 1
        stats = env.get_stats()
        print("=" * 50)
        print(f"Episode {episode}'s stats: ")
        print(stats)
    print("=" * 50)


for t in range(8):
    log_stats()
    env.expand()
    env.clean_stats()
