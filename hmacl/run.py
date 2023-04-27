from collections import abc
import datetime
import os
import random

import numpy as np
import yaml
from types import SimpleNamespace as NameSpace
import torch as th
from config import parser
from hmacl.algo.algo import Algo
from hmacl.cpi import CPI
from hmacl.stg import STG
from hmacl.utils.logging_ import get_logger
from hmacl.algo.components.transforms import OneHot
from hmacl.algo.components.episode_buffer import ReplayBuffer
from hmacl.algo.runners import REGISTRY as r_REGISTRY
from hmacl.algo.controllers import REGISTRY as mac_REGISTRY
from hmacl.algo.learners import REGISTRY as le_REGISTRY
from hmacl.algo.envs import REGISTRY as env_REGISTRY


def main(all_args, _log):
    # start================================args and log========================================
    # seed
    if not hasattr(all_args, "seed"):
        all_args.seed = np.random.randint(0, 100000)
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    th.random.manual_seed(all_args.seed)
    # parallel runner need to setting new seed for envs
    all_args.env_args["seed"] = all_args.seed

    # setting target task env_config
    # all_args.tag_env_args = deepcopy(all_args.env_args)
    # all_args.tag_env_args["map_size"] = all_args.tag_env_args["tag_map_size"]

    fmt_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    all_args.unique_token = f"{all_args.name}_seed{all_args.seed}_{all_args.env}-{all_args.env_args['scenario']}_{fmt_time}"

    if all_args.use_wandb:
        _log.setup_wandb(all_args)
    # end==================================args and log========================================

    # start========================marl algorithm required components==========================
    # Set up schemes and groups here
    env_demo = env_REGISTRY[all_args.env](**all_args.env_args)
    env_info = env_demo.get_env_info()
    all_args.n_agents = env_info["n_agents"]
    all_args.n_actions = env_info["n_actions"]
    all_args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": all_args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=all_args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        all_args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if all_args.buffer_cpu_only else all_args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[all_args.mac](buffer.scheme, groups, all_args)

    # Give runner the scheme
    runner = r_REGISTRY[all_args.runner](args=all_args, logger=_log)

    # Learner
    learner = le_REGISTRY[all_args.learner](mac, buffer.scheme, _log, all_args)

    if all_args.use_cuda:
        learner.cuda()
    # end==========================marl algorithm required components==========================

    # start=================================HMACL components===================================
    # hmacl
    cpi = CPI(all_args, _log)
    stg = STG(all_args, _log)
    algo = Algo(all_args, _log, runner, mac, learner, buffer)

    # initial env/target env
    task_id = 0
    env_config = stg.get_task_config(task_id)
    env = env_REGISTRY[all_args.env](**env_config)
    tag_env_config = stg.get_tag_task_config()
    tag_env = env_REGISTRY[all_args.env](**tag_env_config)
    runner.set_env(env, tag_env)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    cpi.update_stats(algo, learner, 0)
    # end===================================HMACL components===================================

    while runner.t_env < args.t_max:
        td_error = algo.run()
        cpi.improve(algo, learner, runner.t_env)
        task_id = stg.generate(task_id, cpi.cur_metrics, td_error)
        env_config = stg.get_task_config(task_id)
        env.update(env_config)

    # Train on target_task
    algo.run()

    runner.close_env()
    _log.console_logger.info("Finished Training")


def load_config_dict():
    env_config_name = args.env
    algo_config_name = args.name
    with open(os.path.join(os.path.dirname(__file__), "algo", "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    env_config = _get_config(env_config_name, "envs")
    alg_config = _get_config(algo_config_name, "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # merge params and args
    for k, v in args.__dict__.items():
        config_dict[k] = v

    return config_dict


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config

def _get_config(config_name, subfolder):

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "algo", "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == "__main__":
    _log = get_logger()
    args = parser.parse_args()
    config_dict = load_config_dict()
    args_sanity_check(config_dict, _log)
    all_args = NameSpace(**config_dict)
    all_args.device = "cuda" if all_args.use_cuda else "cpu"
    main(all_args, _log)
