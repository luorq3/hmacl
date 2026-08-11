"""Single-instance VMAS adapter for HMACL runners."""

from copy import deepcopy

import numpy as np
import torch

from hmacl.algo.envs.multiagentenv import MultiAgentEnv


class VMASFootballEnv(MultiAgentEnv):
    """Expose discrete VMAS Football through the PyMARL environment API.

    Parallelism is owned by :class:`ParallelRunner`; every wrapper therefore
    creates exactly one VMAS simulation. A curriculum update reconstructs that
    simulation because VMAS fixes team sizes when the world is created.
    """

    def __init__(self, **config):
        self._config = deepcopy(config)
        self._env = None
        self._obs = None
        self._episode_steps = 0
        self._build()

    def _build(self):
        try:
            from vmas import make_env
        except ImportError as exc:
            raise RuntimeError(
                "VMAS Football requires the optional 'vmas' dependency"
            ) from exc

        config = deepcopy(self._config)
        config.pop("target_n_agents", None)
        config.pop("scale_red_team", None)
        if config.get("scenario") != "football":
            raise ValueError("VMASFootballEnv only supports scenario='football'")
        if config.get("continuous_actions", False):
            raise ValueError("HMACL currently requires discrete VMAS actions")
        config["num_envs"] = 1
        config["dict_spaces"] = False
        self._seed = int(config.get("seed", 0))
        self._env = make_env(**config)
        self.n_agents = int(self._env.n_agents)
        self.episode_limit = int(self._env.max_steps)
        self.n_actions = max(int(space.n) for space in self._env.action_space)
        self.obs_shape = max(
            int(np.prod(space.shape)) for space in self._env.observation_space
        )
        self._episode_steps = 0

    def reset(self):
        self._obs = self._convert_obs(self._env.reset())
        self._episode_steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        action_array = np.asarray(actions).reshape(self.n_agents, -1)
        vm_actions = [
            torch.tensor(
                [int(action_array[index, 0])],
                dtype=torch.long,
                device=self._env.device,
            )
            for index in range(self.n_agents)
        ]
        obs, rewards, dones, infos = self._env.step(vm_actions)
        self._obs = self._convert_obs(obs)
        self._episode_steps += 1

        reward = float(
            torch.stack([value.reshape(-1)[0] for value in rewards]).mean().item()
        )
        terminated = bool(dones.reshape(-1)[0].item())
        sparse_reward = float(infos[0]["sparse_reward"].reshape(-1)[0].item())
        episode_limit = self._episode_steps >= self.episode_limit and sparse_reward == 0
        info = {
            "episode_limit": episode_limit,
            "battle_won": float(sparse_reward > 0),
            "battle_lost": float(sparse_reward < 0),
            "win_rate": float(sparse_reward > 0),
        }
        return reward, terminated, info

    def get_obs(self):
        if self._obs is None:
            raise RuntimeError("reset must be called before reading observations")
        return self._obs.copy()

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.obs_shape

    def get_state(self):
        return self.get_obs().reshape(-1)

    def get_state_size(self):
        return self.n_agents * self.obs_shape

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions), dtype=np.int64)

    def get_avail_agent_actions(self, agent_id):
        space = self._env.action_space[agent_id]
        available = np.zeros(self.n_actions, dtype=np.int64)
        available[: int(space.n)] = 1
        return available

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def update(self, config):
        self.close()
        self._config = deepcopy(dict(config))
        self._obs = None
        self._build()

    def get_stats(self):
        return {}

    def render(self):
        return self._env.render(mode="human")

    def close(self):
        close = getattr(self._env, "close", None)
        if close is not None:
            close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
        self._env.seed(self._seed)
        return self._seed

    def save_replay(self):
        return None

    def _convert_obs(self, observations):
        converted = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)
        for index, observation in enumerate(observations):
            values = observation.detach().cpu().numpy().reshape(-1)
            converted[index, : values.size] = values
        return converted
