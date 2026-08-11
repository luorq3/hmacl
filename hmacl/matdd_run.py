"""Run the recovered MATDD algorithm on a supported HMACL environment."""

import datetime
import json
import os
import random
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

from hmacl.config import get_config
from hmacl.matdd.adapters.football import FootballTaskAdapter
from hmacl.matdd.adapters.m2ale import M2ALETaskAdapter
from hmacl.matdd.designer import PPOCurriculumDesigner, RandomCurriculumDesigner
from hmacl.matdd.dispatcher import CurriculumDispatcher
from hmacl.matdd.hmacl_backend import build_hmacl_backend
from hmacl.matdd.loop import MATDDConfig, MATDDTrainingLoop
from hmacl.utils.common import increment_path
from hmacl.utils.config import args_sanity_check, load_config_dict
from hmacl.utils.logging_ import get_logger


def get_parser():
    parser = get_config()
    parser.description = "Recovered MATDD on HMACL"
    parser.set_defaults(
        eval_nepisode=32,
        test_nepisode=32,
        t_max=4000000,
        t_update=50000,
        use_wandb=False,
    )
    parser.add_argument("--designer", choices=("ppo", "random"), default="ppo")
    parser.add_argument("--batch_size_run", type=int, default=1)
    parser.add_argument("--target_agents", type=int, default=None)
    parser.add_argument("--episode_limit", type=int, default=None)
    parser.add_argument(
        "--student_architecture",
        choices=("auto", "paper", "legacy"),
        default="auto",
    )
    parser.add_argument("--matdd_encoder_dim", type=int, default=256)
    parser.add_argument("--matdd_recurrent_dim", type=int, default=256)
    parser.add_argument("--matdd_projection_dim", type=int, default=128)
    parser.add_argument("--student_gamma", type=float, default=None)
    parser.add_argument("--student_grad_norm_clip", type=float, default=None)
    parser.add_argument("--curriculum_iterations", type=int, default=80)
    parser.add_argument("--steps_per_curriculum", type=int, default=50000)
    parser.add_argument("--final_target_steps", type=int, default=50000)
    parser.add_argument("--curriculum_eval_episodes", type=int, default=5)
    parser.add_argument("--dispatcher_capacity", type=int, default=8)
    parser.add_argument("--dispatcher_rho", type=float, default=0.5)
    parser.add_argument("--dispatcher_temperature", type=float, default=0.1)
    parser.add_argument("--teacher_update_interval", type=int, default=8)
    parser.add_argument("--teacher_hidden_dim", type=int, default=128)
    parser.add_argument("--teacher_lr", type=float, default=1e-4)
    parser.add_argument("--teacher_gamma", type=float, default=0.995)
    parser.add_argument("--teacher_gae_lambda", type=float, default=0.95)
    parser.add_argument("--teacher_epochs", type=int, default=4)
    parser.add_argument("--teacher_minibatch_size", type=int, default=8)
    parser.add_argument("--teacher_return_scale", type=float, default=100.0)
    return parser


def main():
    parser = get_parser()
    cli_args = parser.parse_args()
    logger = get_logger("matdd")
    config_dict = load_config_dict(cli_args)
    args_sanity_check(config_dict, logger)
    args = SimpleNamespace(**config_dict)
    if args.student_architecture == "paper" or (
        args.student_architecture == "auto" and args.env == "vmas"
    ):
        args.agent = "matdd_rnn"
    if args.student_gamma is not None:
        args.gamma = args.student_gamma
    if args.student_grad_norm_clip is not None:
        args.grad_norm_clip = args.student_grad_norm_clip
    args.device = "cuda" if args.use_cuda else "cpu"
    _seed_everything(args.seed)
    _prepare_run_directory(args)
    if args.use_wandb:
        logger.setup_wandb(args)

    adapter = _build_adapter(args)
    protagonist = build_hmacl_backend(
        args, logger, adapter, "protagonist", seed_offset=0
    )
    antagonist = build_hmacl_backend(
        args, logger, adapter, "antagonist", seed_offset=10000
    )
    dispatcher = CurriculumDispatcher(
        adapter.space,
        adapter.target_task,
        capacity=args.dispatcher_capacity,
        rho=args.dispatcher_rho,
        temperature=args.dispatcher_temperature,
        seed=args.seed,
    )
    designer = _build_designer(args, adapter)
    loop = MATDDTrainingLoop(
        protagonist=protagonist,
        antagonist=antagonist,
        designer=designer,
        dispatcher=dispatcher,
        initial_task=adapter.initial_task,
        target_task=adapter.target_task,
        config=MATDDConfig(
            curriculum_iterations=args.curriculum_iterations,
            steps_per_curriculum=args.steps_per_curriculum,
            final_target_steps=args.final_target_steps,
            evaluation_episodes=args.eval_nepisode,
            teacher_update_interval=args.teacher_update_interval,
            seed=args.seed,
        ),
    )
    try:
        result = loop.run()
        _save_result(args, result, dispatcher)
        protagonist.save(os.path.join(args.local_results_path, "models", "protagonist"))
        antagonist.save(os.path.join(args.local_results_path, "models", "antagonist"))
        designer.save(os.path.join(args.local_results_path, "models", "teacher.th"))
        logger.console_logger.info(
            "MATDD finished with %s total environment steps",
            result.total_environment_steps,
        )
    finally:
        protagonist.close()
        antagonist.close()


def _build_designer(args, adapter):
    if args.designer == "random":
        return RandomCurriculumDesigner(
            adapter.space, adapter.target_task, seed=args.seed
        )
    return PPOCurriculumDesigner(
        adapter.space,
        adapter.target_task,
        hidden_dim=args.teacher_hidden_dim,
        learning_rate=args.teacher_lr,
        gamma=args.teacher_gamma,
        gae_lambda=args.teacher_gae_lambda,
        epochs=args.teacher_epochs,
        minibatch_size=args.teacher_minibatch_size,
        return_scale=args.teacher_return_scale,
        device=args.device,
        seed=args.seed,
    )


def _build_adapter(args):
    if args.env == "m2ale":
        config = dict(args.env_args)
        if args.episode_limit is not None:
            config["time_limit"] = args.episode_limit
        args.env_args = config
        return M2ALETaskAdapter(config)
    if args.env == "vmas":
        config = dict(args.env_args)
        if args.target_agents is not None:
            config["target_n_agents"] = args.target_agents
        if args.episode_limit is not None:
            config["max_steps"] = args.episode_limit
        args.env_args = config
        return FootballTaskAdapter(config)
    raise ValueError("MATDD supports env='m2ale' or env='vmas'")


def _prepare_run_directory(args):
    base = os.path.abspath(args.local_results_path)
    args.local_results_path = str(
        increment_path(base, args.exp, exist_ok=args.exp_exist_ok, mkdir=True)
    )
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    args.unique_token = "matdd_seed{}_{}".format(args.seed, timestamp)


def _save_result(args, result, dispatcher):
    path = os.path.join(args.local_results_path, "matdd_result.json")
    payload = {
        "config": _json_safe(vars(args)),
        "total_environment_steps": result.total_environment_steps,
        "iterations": [_json_safe(asdict(item)) for item in result.iterations],
        "final_training": (
            _json_safe(asdict(result.final_training)) if result.final_training else None
        ),
        "dispatcher": _json_safe(dispatcher.snapshot()),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
