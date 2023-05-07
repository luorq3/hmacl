import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import datetime
import random
import numpy as np
from types import SimpleNamespace as NameSpace
import torch as th
from config import get_config
from hmacl.algo.algo import Algo
from hmacl.cpi import CPI
from hmacl.stg import STG
from hmacl.utils.common import increment_path
from hmacl.utils.config import args_sanity_check, load_config_dict
from hmacl.utils.logging_ import get_logger
from hmacl.algo.components.transforms import OneHot
from hmacl.algo.components.episode_buffer import ReplayBuffer
from hmacl.algo.runners import REGISTRY as r_REGISTRY
from hmacl.algo.controllers import REGISTRY as mac_REGISTRY
from hmacl.algo.learners import REGISTRY as le_REGISTRY
from hmacl.algo.envs import REGISTRY as env_REGISTRY


def main(all_args, _log, tuning=False):
    # start================================args and log========================================
    # ray tune
    if tuning:
        from ray.air import session

    # seed
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    th.random.manual_seed(all_args.seed)
    # parallel runner need to setting new seed for envs
    all_args.env_args["seed"] = all_args.seed

    # if obs appending one hot agent id
    if all_args.obs_agent_id:
        all_args.env_args["obs_with_agent_id"] = True
        all_args.obs_agent_id = False

    # result path and exp_name
    local_results_path = os.path.abspath(all_args.local_results_path)
    exp_prefix = 'exp' if all_args.exp is None else all_args.exp
    all_args.local_results_path = increment_path(local_results_path, exp_prefix, exist_ok=all_args.exp_exist_ok, mkdir=True)
    all_args.exp = os.path.basename(all_args.local_results_path)

    fmt_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    all_args.unique_token = f"seed{all_args.seed}_{fmt_time}"

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

    _log_prefix = "hmacl_"
    _log.log_stat(_log_prefix + 'task', task_id, runner.t_env)
    _log.log_stat(_log_prefix + 'map_size_X', env_config['map_size'][0], runner.t_env)

    n_hmacl_episode = all_args.t_max // all_args.t_update
    hmacl_episode = 0

    _log.console_logger.info(f"Hmacl episode starting, generated task: {task_id}.")
    while hmacl_episode < n_hmacl_episode:
        td_error = algo.run()
        _log.log_stat(_log_prefix + 'td_error', td_error, runner.t_env)

        cpi.improve(algo, learner, runner.t_env)
        if tuning:
            session.report({all_args.metrics: cpi.cur_metrics})
        task_id = stg.generate(task_id, cpi.cur_metrics, td_error)
        env_config = stg.get_task_config(task_id)
        env.update(env_config)

        _log.log_stat(_log_prefix + 'task', task_id, runner.t_env)
        _log.log_stat(_log_prefix + 'map_size_X', env_config['map_size'][0], runner.t_env)
        _log.console_logger.info(f"Hmacl episode: {hmacl_episode} completed, generated task: {task_id}.")

        hmacl_episode += 1

    _log.console_logger.info(f"Hmacl episode ended, count for hmacl_episode: {hmacl_episode}.")
    # Train on target_task
    env.update(tag_env_config)
    _log.console_logger.info(f"Hmacl turning started, tag_task: {tag_env_config['map_size']}.")
    td_error = algo.run()
    cpi.save_model(learner, "latest")
    _log.console_logger.info(f"Hmacl turning ended, td_error: {td_error}.")

    runner.close_env()
    _log.console_logger.info("Finished Training")


def search(config):
    _log = get_logger(f"proc-seed{config['seed']}")
    parser = get_config()
    args, unknow = parser.parse_known_args()
    for k, v in config.items():
        setattr(args, k, v)
    config_dict = load_config_dict(args)
    args_sanity_check(config_dict, _log)
    all_args = NameSpace(**config_dict)
    all_args.device = "cuda" if all_args.use_cuda else "cpu"
    main(all_args, _log, tuning=True)


if __name__ == "__main__":
    _log = get_logger()
    parser = get_config()
    args = parser.parse_args()
    config_dict = load_config_dict(args)
    args_sanity_check(config_dict, _log)
    all_args = NameSpace(**config_dict)
    all_args.device = "cuda" if all_args.use_cuda else "cpu"
    main(all_args, _log)
