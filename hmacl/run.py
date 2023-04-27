from collections import abc
import datetime
import os
import random
from copy import deepcopy

import numpy as np
import yaml
from types import SimpleNamespace as NameSpace
import torch as th
from config import parser
from hmacl.algo.algo import Algo
from hmacl.cpi import CPI
from hmacl.stg import STG
from hmacl.utils.logging_ import get_logger


def main(all_args, _log):
    # seed
    if not hasattr(all_args, "seed"):
        all_args.seed = np.random.randint(0, 100000)
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    th.random.manual_seed(all_args.seed)
    # parallel runner need to setting new seed for envs
    all_args.env_args["seed"] = all_args.seed

    # setting target task env_config
    all_args.tag_env_args = deepcopy(all_args.env_args)
    all_args.tag_env_args["map_size"] = all_args.tag_env_args["tag_map_size"]

    fmt_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    all_args.unique_token = f"{all_args.name}_seed{all_args.seed}_{all_args.env}-{all_args.env_args['scenario']}_{fmt_time}"

    if all_args.use_wandb:
        _log.setup_wandb(all_args)

    algo = Algo(all_args, _log)

    metrics = algo.eval_tag_task()
    _log.console_logger.info("Evaluate target task in initial policies, stats: {}".format(metrics))

    cpi = CPI(all_args.target_task, all_args.ps_type, all_args.update_type, **all_args)
    stg = STG(all_args.env, all_args.scenario, **all_args)

    n_timesteps = all_args.n_timesteps
    t = 0
    # steps_per_algo
    steps_per_algo = None
    while t < n_timesteps:
        loss = algo.run()
        metrics = cpi.improve()
        next_task = stg.generate(cur_task, metrics, loss)
        cur_task = next_task
        t += steps_per_algo

    # Train on target_task
    algo.run()


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
