"""Bridge between the recovered MATDD loop and the existing HMACL stack."""

import os
import random
from copy import deepcopy
from typing import Mapping

import numpy as np

from hmacl.algo.algo import Algo
from hmacl.algo.components.episode_buffer import ReplayBuffer
from hmacl.algo.components.transforms import OneHot
from hmacl.algo.controllers import REGISTRY as mac_REGISTRY
from hmacl.algo.envs import REGISTRY as env_REGISTRY
from hmacl.algo.learners import REGISTRY as learner_REGISTRY
from hmacl.algo.runners import REGISTRY as runner_REGISTRY

from .adapters.padded_env import PaddedMultiAgentEnv
from .loop import EvaluationResult, TrainingResult
from .task_space import Number


class RoleLogger:
    """Prefix scalar names while sharing one console and experiment logger."""

    def __init__(self, logger, role):
        self._logger = logger
        self.role = role
        self.console_logger = logger.console_logger

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def log_stat(self, key, value, t):
        if key == "episode":
            self._logger.log_stat(key, value, t)
        else:
            self._logger.log_stat("{}/{}".format(self.role, key), value, t)


class HMACLStudentBackend:
    """Run one independently initialized HMACL learner for MATDD."""

    def __init__(self, algo, adapter, current_evaluation_episodes=5, seed=0):
        if current_evaluation_episodes <= 0:
            raise ValueError("current_evaluation_episodes must be positive")
        self.algo = algo
        self.runner = algo.runner
        self.learner = algo.learner
        self.adapter = adapter
        self.current_evaluation_episodes = current_evaluation_episodes
        self.current_task = dict(adapter.initial_task)
        self.seed = int(seed)
        self.task_updates = 0

    def train(self, task: Mapping[str, Number], step_budget: int) -> TrainingResult:
        self._set_task(task)
        self.algo.args.t_update = step_budget
        before = self.runner.t_env
        td_error = float(self.algo.run())
        training_steps = self.runner.t_env - before
        current_evaluation = self._evaluate(episodes=self.current_evaluation_episodes)
        return TrainingResult(
            mean_return=current_evaluation.mean_return,
            mean_abs_td_error=td_error,
            environment_steps=training_steps + current_evaluation.environment_steps,
            metrics=current_evaluation.metrics,
        )

    def evaluate(self, task: Mapping[str, Number], episodes: int) -> EvaluationResult:
        if self.adapter.space.key(task) != self.adapter.space.key(
            self.adapter.target_task
        ):
            raise ValueError(
                "HMACL target evaluation only supports the configured target task"
            )
        previous_task = dict(self.current_task)
        self._set_task(task)
        try:
            return self._evaluate(episodes=episodes)
        finally:
            self._set_task(previous_task)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.learner.save_models(path)

    def close(self):
        self.runner.close_env()

    def _set_task(self, task):
        config = self.adapter.env_config(task)
        self.task_updates += 1
        config["seed"] = self.seed + self.task_updates * self.runner.batch_size
        self.runner.update_env(config)
        self.current_task = dict(task)

    def _evaluate(self, episodes):
        returns = []
        infos = []
        environment_steps = 0
        runs = (episodes + self.runner.batch_size - 1) // self.runner.batch_size
        for _ in range(runs):
            self.runner.run(test_mode=True)
            returns.extend(self.runner.last_episode_returns)
            infos.extend(self.runner.last_episode_infos)
            environment_steps += self.runner.last_run_environment_steps
        mean_return = sum(returns) / len(returns)
        metrics = _mean_numeric_metrics(infos)
        metrics.update(
            {
                "return_std": _standard_deviation(returns),
                "evaluation_episodes": len(returns),
            }
        )
        return EvaluationResult(
            mean_return=mean_return,
            environment_steps=environment_steps,
            metrics=metrics,
        )


def build_hmacl_backend(all_args, logger, adapter, role, seed_offset=0):
    """Construct an independent learner, environment, buffer, and runner."""

    args = deepcopy(all_args)
    args.seed += seed_offset
    args.env_args = deepcopy(args.env_args)
    args.env_args["seed"] = args.seed
    if args.runner not in {"episode", "parallel"}:
        raise ValueError("MATDD HMACL backend supports episode or parallel runners")
    if args.runner == "episode":
        args.batch_size_run = 1
    args.enable_periodic_test = False
    args.enable_periodic_target_eval = False
    args.checkpoint_path = ""
    args.unique_token = "{}_{}".format(all_args.unique_token, role)
    role_logger = RoleLogger(logger, role)
    _seed_backend(args.seed)

    initial_config = adapter.env_config(adapter.initial_task)
    initial_config["seed"] = args.seed
    target_config = adapter.env_config(adapter.target_task)
    target_config["seed"] = args.seed + 1
    raw_target_env = env_REGISTRY[args.env](**target_config)
    env_info = raw_target_env.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_shape = env_info["obs_shape"]
    args.state_shape = env_info["state_shape"]
    if getattr(args, "cps", False):
        args.ap2cp = raw_target_env.get_classes()
    raw_target_env.close()

    env = None
    if args.runner == "episode":
        env = PaddedMultiAgentEnv(env_REGISTRY[args.env](**initial_config), env_info)
    else:
        args.env_args = initial_config
        args.matdd_target_env_info = deepcopy(env_info)

    torch = _torch()
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "agent_mask": {"vshape": (1,), "group": "agents"},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner = runner_REGISTRY[args.runner](args=args, logger=role_logger)
    if env is not None:
        runner.set_env(env)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = learner_REGISTRY[args.learner](mac, buffer.scheme, role_logger, args)
    if args.use_cuda:
        learner.cuda()
    algo = Algo(args, role_logger, runner, mac, learner, buffer)
    return HMACLStudentBackend(
        algo,
        adapter,
        current_evaluation_episodes=getattr(args, "curriculum_eval_episodes", 5),
        seed=args.seed,
    )


def _standard_deviation(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _mean_numeric_metrics(infos):
    values = {}
    for info in infos:
        for key, value in info.items():
            if key == "episode_limit" or not isinstance(
                value, (bool, int, float, np.number)
            ):
                continue
            values.setdefault(key, []).append(float(value))
    return {
        key: sum(metric_values) / len(metric_values)
        for key, metric_values in values.items()
        if metric_values
    }


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("HMACL backend requires PyTorch") from exc
    return torch


def _seed_backend(seed):
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
