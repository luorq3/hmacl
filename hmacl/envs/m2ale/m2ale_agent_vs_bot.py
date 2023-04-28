import random

import numpy as np

from argparse import Namespace

from hmacl.envs.m2ale.multiagentenv import MultiAgentEnv
from hmacl.envs.m2ale.m2ale_core import M2ALECore


class M2ALE(MultiAgentEnv):
    def __init__(self, **kwargs):
        config = Namespace(**kwargs)
        super(M2ALE, self).__init__()
        self.camp = config.camp
        self.key = config.scenario
        self.episode_limit = config.time_limit

        # stats
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0

        self._env = M2ALECore(config)
        self._init()

    # get mapping from agent to type
    def get_agents_types(self):
        agents_types = {}
        for agent_id, unique_id in self.agent_idx_dict.items():
            unit = self.get_unit_by_unique_id(unique_id)
            agents_types[agent_id] = unit.agent_type

        return agents_types

    def get_unit_by_unique_id(self, unique_id):
        return self._env.units_dict.get(unique_id)

    def _init(self):
        self._episode_steps = 0

        # self.agent_idx_dict = {agent_id: unique_id for agent_id, unique_id in enumerate(sorted(self._env.agents_unique_idx))}
        self.agent_idx_dict = {}
        action_dim = 0
        for agent_id, unique_id in enumerate(sorted(self._env.agents_unique_idx)):
            agent = self._env.get_agent(unique_id)
            agent.agent_id = agent_id
            action_dim = max(agent.action_dim, action_dim)
            self.agent_idx_dict[agent_id] = unique_id
        self.action_dim = action_dim
        self.n_agents = len(self.agent_idx_dict)
        self.reward_type = self._env.reward_type

    def step(self, actions):
        action_int = [int(a) for a in actions]

        new_info = {}

        action_dict = {agent_id: action for agent_id, action in zip(self.agent_idx_dict.values(), action_int)}
        rewards, done, info = self._env.step(action_dict)

        self._total_steps += 1
        self._episode_steps += 1

        reward = 0
        if done:
            new_info["battle_won"] = False
            self.battles_game += 1
            if info["of_win"]:
                if info["de_win"]:
                    self.timeouts += 1
                else:
                    self.battles_won += 1
                    new_info["battle_won"] = True
                    if self.reward_type == "sparse":
                        reward = 1
            elif info["de_win"]:
                if self.reward_type == "sparse":
                    reward = -1
        elif self._episode_steps >= self.episode_limit:
            done = True
            new_info["episode_limit"] = True
            new_info["battle_won"] = False
            self.battles_game += 1
            self.timeouts += 1

        if done:
            self._episode_count += 1

        if self.reward_type == "dense":
            reward = sum(rewards.values())
        return reward, done, new_info

    def get_obs(self):
        obs = []
        obs_dict = self._env.get_obs()
        for unique_id in self.agent_idx_dict.values():
            obs.append(obs_dict.get(unique_id))
        return np.array(obs, dtype=np.float32)

    def get_obs_agent(self, agent_id):
        return self._env.get_obs_unique_id(self.agent_idx_dict.get(agent_id))

    def get_obs_size(self):
        obs_dims = []
        obs_dim_dict = self._env.get_obs_dim()
        for unique_id in self.agent_idx_dict.values():
            obs_dims.append(obs_dim_dict.get(unique_id))
        return max(obs_dims)

    def get_state(self):
        return self._env.get_state()

    def get_state_size(self):
        return self._env.state_dims[self.camp]

    def get_avail_actions(self):
        avail_acts = np.zeros((self.n_agents, self.action_dim))
        avail_acts_dict = self._env.get_avail_actions()
        for i, unique_id in enumerate(self.agent_idx_dict.values()):
            acts = avail_acts_dict.get(unique_id)
            avail_acts[i, :len(acts)] = acts
        return avail_acts

    def get_avail_agent_actions(self, agent_id):
        new_act = np.zeros(self.action_dim)
        act = self._env.get_agent_avail_actions(self.agent_idx_dict.get(agent_id))
        new_act[:len(act)] = act
        return new_act

    def get_total_actions(self):
        return self.action_dim

    def reset(self):
        self._env.reset()
        self._init()

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": (self.battles_won / self.battles_game) if self.battles_game != 0 else 0,
            "timeouts": self.timeouts,
            "draw_rate": (self.timeouts / self.battles_game) if self.battles_game != 0 else 0,
            "loss_rate": ((self.battles_game - self.battles_won) / self.battles_game) if self.battles_game != 0 else 0,
        }
        return stats

    # for cpi eval
    def clean_stats(self):
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0

    def update(self, config):
        config = Namespace(**config)
        self.camp = config.camp
        self.key = config.scenario
        self.episode_limit = config.time_limit

        self._env = M2ALECore(config)
        self._init()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = 1
        random.seed(seed)
        np.random.seed(seed)

    def save_replay(self):
        pass
