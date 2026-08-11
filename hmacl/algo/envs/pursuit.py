"""PettingZoo SISL Pursuit adapter for the HMACL environment API."""

from copy import deepcopy

import numpy as np

from hmacl.algo.envs.multiagentenv import MultiAgentEnv


class PettingZooPursuitEnv(MultiAgentEnv):
    """Wrap the current PettingZoo parallel Pursuit environment.

    PettingZoo fixes the map and population at construction, so curriculum
    updates rebuild the environment between complete episodes.
    """

    _ADAPTER_ONLY_KEYS = {
        "target_map_size",
        "target_n_pursuers",
        "scale_n_evaders",
    }

    def __init__(self, **config):
        self._config = deepcopy(config)
        self._env = None
        self._obs = None
        self._episode_steps = 0
        self._build()

    def _build(self):
        try:
            from pettingzoo.sisl import pursuit_v4
        except ImportError as exc:
            raise RuntimeError(
                "Pursuit requires pettingzoo, pygame-ce, pymunk, and scipy"
            ) from exc

        config = deepcopy(self._config)
        for key in self._ADAPTER_ONLY_KEYS:
            config.pop(key, None)
        config.pop("scenario", None)
        config.pop("map_name", None)
        self._seed = int(config.pop("seed", 0))
        self.episode_limit = int(config["max_cycles"])
        self._env = pursuit_v4.parallel_env(**config)
        self.agents = list(self._env.possible_agents)
        self.n_agents = len(self.agents)
        sample_agent = self.agents[0]
        self.obs_shape = tuple(self._env.observation_space(sample_agent).shape)
        self.n_actions = int(self._env.action_space(sample_agent).n)
        state = self._read_state()
        self.state_shape = tuple(state.shape)
        self._episode_steps = 0

    def reset(self):
        observations, _ = self._env.reset(seed=self._seed)
        self._obs = self._ordered_observations(observations)
        self._episode_steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        action_array = np.asarray(actions).reshape(self.n_agents, -1)
        action_dict = {
            agent: int(action_array[index, 0])
            for index, agent in enumerate(self.agents)
            if agent in self._env.agents
        }
        observations, rewards, terminations, truncations, _ = self._env.step(
            action_dict
        )
        if observations:
            self._obs = self._ordered_observations(observations)
        self._episode_steps += 1

        reward_values = [float(rewards.get(agent, 0.0)) for agent in self.agents]
        reward = float(np.mean(reward_values))
        natural_termination = bool(terminations) and all(terminations.values())
        time_limit = bool(truncations) and all(truncations.values())
        terminated = natural_termination or time_limit or not self._env.agents
        base = self._base_environment()
        evaders_gone = getattr(base, "evaders_gone", np.zeros(0, dtype=bool))
        info = {
            "episode_limit": time_limit and not natural_termination,
            "completion_rate": float(natural_termination),
            "evaders_caught": float(np.asarray(evaders_gone).sum()),
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
        return self._read_state()

    def get_state_size(self):
        return self.state_shape

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions), dtype=np.int64)

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.n_actions, dtype=np.int64)

    def get_total_actions(self):
        return self.n_actions

    def update(self, config):
        self.close()
        self._config = deepcopy(dict(config))
        self._obs = None
        self._build()

    def get_stats(self):
        return {}

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
        return self._seed

    def save_replay(self):
        return None

    def _ordered_observations(self, observations):
        values = []
        for agent in self.agents:
            observation = observations.get(agent)
            if observation is None:
                observation = self._env.unwrapped.observe(agent)
            values.append(np.asarray(observation, dtype=np.float32))
        return np.stack(values, axis=0)

    def _base_environment(self):
        raw = self._env.unwrapped
        return getattr(raw, "env", raw)

    def _read_state(self):
        base = self._base_environment()
        model_state = np.asarray(base.model_state[:3], dtype=np.float32)
        return np.moveaxis(model_state, 0, -1).copy()
