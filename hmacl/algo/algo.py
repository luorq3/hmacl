import os
import time
import yaml
import torch as th
from utils.timehelper import time_left, time_str
import collections
from types import SimpleNamespace as sn

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from hmacl.utils.logging_ import get_logger


class Algo:

    def __init__(self, env_config_name, algo_config_name, params):
        with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)
        env_config = _get_config(env_config_name, "envs")
        alg_config = _get_config(algo_config_name, "algs")
        config_dict = recursive_dict_update(config_dict, env_config)
        config_dict = recursive_dict_update(config_dict, alg_config)

        # merge params and args
        for k, v in params.items():
            config_dict[k] = v

        self.logger = get_logger()
        args_sanity_check(config_dict, self.logger)
        args = sn(**config_dict)
        args.device = "cuda" if args.use_cuda else "cpu"

        self.args = args

    def run(self):
        # Init runner so we can get env info
        runner = r_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

        # Set up schemes and groups here
        env_info = runner.get_env_info()
        self.args.n_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]

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
        groups = {"agents": self.args.n_agents}
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])}

        buffer = ReplayBuffer(
            scheme,
            groups,
            self.args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if self.args.buffer_cpu_only else self.args.device,
        )

        # Setup multiagent controller here
        mac = mac_REGISTRY[self.args.mac](buffer.scheme, groups, self.args)

        # Give runner the scheme
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

        # Learner
        learner = le_REGISTRY[self.args.learner](mac, buffer.scheme, self.logger, self.args)

        if self.args.use_cuda:
            learner.cuda()

        if self.args.checkpoint_path != "":

            timesteps = []
            timestep_to_load = 0

            if not os.path.isdir(self.args.checkpoint_path):
                self.logger.console_logger.info(
                    "Checkpoint directiory {} doesn't exist".format(self.args.checkpoint_path)
                )
                return

            # Go through all files in args.checkpoint_path
            for name in os.listdir(self.args.checkpoint_path):
                full_name = os.path.join(self.args.checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if self.args.load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - self.args.load_step))

            model_path = os.path.join(self.args.checkpoint_path, str(timestep_to_load))

            self.logger.console_logger.info("Loading model from {}".format(model_path))
            learner.load_models(model_path)
            runner.t_env = timestep_to_load

            if self.args.evaluate or self.args.save_replay:
                runner.log_train_stats_t = runner.t_env
                self.evaluate_sequential(runner)
                self.logger.log_stat("episode", runner.t_env, runner.t_env)
                self.logger.print_recent_stats()
                self.logger.console_logger.info("Finished Evaluation")
                return

        # start training
        episode = 0
        last_test_T = -self.args.test_interval - 1
        last_eval_T = -self.args.eval_interval - 1
        last_log_T = 0
        model_save_time = 0

        start_time = time.time()
        last_time = start_time

        self.logger.info("Beginning training for {} timesteps".format(self.args.t_max))

        while runner.t_env <= self.args.t_max:

            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(self.args.batch_size):
                episode_sample = buffer.sample(self.args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != self.args.device:
                    episode_sample.to(self.args.device)

                learner.train(episode_sample, runner.t_env, episode)

            # Execute test runs once in a while
            n_test_runs = max(1, self.args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / self.args.test_interval >= 1.0:

                self.logger.console_logger.info(
                    "t_env: {} / {}".format(runner.t_env, self.args.t_max)
                )
                self.logger.console_logger.info(
                    "Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env, self.args.t_max),
                        time_str(time.time() - start_time),
                    )
                )
                last_time = time.time()

                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

            # Execute evaluate in target task once in a while
            n_eval_runs = max(1, self.args.eval_nepisode // runner.batch_size)
            if (runner.t_env - last_eval_T) / self.args.eval_interval >= 1.0:
                last_eval_T = runner.t_env
                for _ in range(n_eval_runs):
                    runner.run(test_mode=True, test_tag=True)

            if self.args.save_model and (
                    runner.t_env - model_save_time >= self.args.save_model_interval
                    or model_save_time == 0
            ):
                model_save_time = runner.t_env
                save_path = os.path.join(
                    self.args.local_results_path, "models", self.args.unique_token, str(runner.t_env)
                )
                # "results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                self.logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

            episode += self.args.batch_size_run

            if (runner.t_env - last_log_T) >= self.args.log_interval:
                self.logger.log_stat("episode", episode, runner.t_env)
                self.logger.print_recent_stats()
                last_log_T = runner.t_env

        runner.close_env()
        self.logger.console_logger.info("Finished Training")

    def evaluate_sequential(self, runner):

        for _ in range(self.args.test_nepisode):
            runner.run(test_mode=True)

        if self.args.save_replay:
            runner.save_replay()

        runner.close_env()


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
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
