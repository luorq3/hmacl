"""Fixed-shape facade for curricula with varying agent and feature counts."""

from copy import deepcopy

import numpy as np


class PaddedMultiAgentEnv:
    """Present every curriculum with the target task's interface dimensions.

    Real agents occupy the leading rows. Missing agents and feature dimensions
    are zero-padded; dummy agents can take only the configured no-op action.
    The wrapped environment must never exceed the target dimensions.
    """

    def __init__(self, env, target_env_info, no_op_action=0):
        self.env = env
        self.target_env_info = deepcopy(dict(target_env_info))
        self.n_agents = int(target_env_info["n_agents"])
        self.n_actions = int(target_env_info["n_actions"])
        self.obs_shape = _shape_tuple(target_env_info["obs_shape"])
        self.state_shape = _shape_tuple(target_env_info["state_shape"])
        self.episode_limit = int(target_env_info["episode_limit"])
        self.no_op_action = int(no_op_action)
        if not 0 <= self.no_op_action < self.n_actions:
            raise ValueError("no-op action is outside the target action space")
        self._validate_current_env()

    @property
    def current_n_agents(self):
        return int(self.env.n_agents)

    def get_env_info(self):
        return deepcopy(self.target_env_info)

    def get_obs(self):
        current = np.asarray(self.env.get_obs())
        target = np.zeros((self.n_agents, *self.obs_shape), dtype=current.dtype)
        _copy_prefix(current, target)
        return target

    def get_obs_agent(self, agent_id):
        if agent_id < self.current_n_agents:
            current = np.asarray(self.env.get_obs_agent(agent_id))
            target = np.zeros(self.obs_shape, dtype=current.dtype)
            _copy_prefix(current, target)
            return target
        return np.zeros(self.obs_shape, dtype=np.float32)

    def get_obs_size(self):
        return _restore_scalar_shape(self.obs_shape)

    def get_state(self):
        current = np.asarray(self.env.get_state())
        target = np.zeros(self.state_shape, dtype=current.dtype)
        _copy_prefix(current, target)
        return target

    def get_state_size(self):
        return _restore_scalar_shape(self.state_shape)

    def get_avail_actions(self):
        current = np.asarray(self.env.get_avail_actions())
        target = np.zeros((self.n_agents, self.n_actions), dtype=current.dtype)
        _copy_prefix(current, target)
        target[self.current_n_agents :, self.no_op_action] = 1
        return target

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_agent_mask(self):
        mask = np.zeros((self.n_agents, 1), dtype=np.float32)
        mask[: self.current_n_agents] = 1.0
        return mask

    def get_total_actions(self):
        return self.n_actions

    def get_classes(self):
        current = np.asarray(self.env.get_classes())
        target = np.full(self.n_agents, -1, dtype=current.dtype)
        target[: self.current_n_agents] = current
        return target

    def step(self, actions):
        return self.env.step(actions[: self.current_n_agents])

    def reset(self):
        result = self.env.reset()
        self._validate_current_env()
        return result

    def update(self, config):
        self.env.update(config)
        self._validate_current_env()

    def get_stats(self):
        return self.env.get_stats()

    def clean_stats(self):
        if hasattr(self.env, "clean_stats"):
            return self.env.clean_stats()
        return None

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def save_replay(self):
        return self.env.save_replay()

    def _validate_current_env(self):
        info = self.env.get_env_info()
        if int(info["n_agents"]) > self.n_agents:
            raise ValueError("curriculum has more agents than the target task")
        if int(info["n_actions"]) > self.n_actions:
            raise ValueError("curriculum has more actions than the target task")
        if not _fits(_shape_tuple(info["obs_shape"]), self.obs_shape):
            raise ValueError("curriculum observation shape exceeds the target task")
        if not _fits(_shape_tuple(info["state_shape"]), self.state_shape):
            raise ValueError("curriculum state shape exceeds the target task")
        if int(info["episode_limit"]) > self.episode_limit:
            raise ValueError("curriculum episode limit exceeds the target task")


def _shape_tuple(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)


def _restore_scalar_shape(shape):
    return shape[0] if len(shape) == 1 else shape


def _fits(current, target):
    return len(current) == len(target) and all(
        current_size <= target_size
        for current_size, target_size in zip(current, target)
    )


def _copy_prefix(source, target):
    if source.ndim != target.ndim:
        raise ValueError(
            "cannot pad an array with shape {} to {}".format(source.shape, target.shape)
        )
    slices = tuple(
        slice(0, min(source_size, target_size))
        for source_size, target_size in zip(source.shape, target.shape)
    )
    target[slices] = source[slices]
