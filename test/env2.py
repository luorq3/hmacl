import numpy as np
from argparse import ArgumentParser
from hmacl.envs.m2ale.m2ale_agent_vs_bot import M2ALE

parser = ArgumentParser()
parser.add_argument("--scenario", type=str, default="7u_vs_10u")
parser.add_argument("--camp", type=str, default="offensive")
parser.add_argument("--reward_type", type=str, default="sparse")
parser.add_argument("--time_limit", type=int, default=200)
parser.add_argument("--map_size", type=list, default=[10, 10])
parser.add_argument("--expand_degree", type=int, default=5)
config = parser.parse_args()

env = M2ALE(**config.__dict__)
env.reset()
num_agents = 5

env.reset()

obs = env.get_obs()
state = env.get_state()
avail_acts = env.get_avail_actions()
print(obs.shape)
print(state.shape)
print(avail_acts.shape)
print(env.get_agents_types())

