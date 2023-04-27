import os
import time
import torch as th
from hmacl.algo.utils.timehelper import time_left, time_str

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


class Algo:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        # Init runner so we can get env info
        self.runner = r_REGISTRY[self.args.runner](args=self.args, logger=self.logger)
        # Set up schemes and groups here
        env_info = self.runner.get_env_info()
        self.args.n_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]

        # Default/Base scheme
        self.scheme = {
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
        self.groups = {"agents": self.args.n_agents}
        self.preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])}

        self.buffer = ReplayBuffer(
            self.scheme,
            self.groups,
            self.args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=self.preprocess,
            device="cpu" if self.args.buffer_cpu_only else self.args.device,
        )

        # Setup multiagent controller here
        self.mac = mac_REGISTRY[self.args.mac](self.buffer.scheme, self.groups, self.args)

        # Give runner the scheme
        self.runner.setup(scheme=self.scheme, groups=self.groups, preprocess=self.preprocess, mac=self.mac)

        # Learner
        self.learner = le_REGISTRY[self.args.learner](self.mac, self.buffer.scheme, self.logger, self.args)

        if self.args.use_cuda:
            self.learner.cuda()

        if self.args.checkpoint_path != "":

            timesteps = []

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
            self.learner.load_models(model_path)
            self.runner.t_env = timestep_to_load

            if self.args.evaluate or self.args.save_replay:
                self.runner.log_train_stats_t = self.runner.t_env
                self.evaluate_sequential()
                self.logger.log_stat("episode", self.runner.t_env, self.runner.t_env)
                self.logger.print_recent_stats()
                self.logger.console_logger.info("Finished Evaluation")
                return

        # start training
        self.episode = 0
        self.last_test_T = -self.args.test_interval - 1
        self.last_eval_T = -self.args.eval_interval - 1
        self.last_log_T = 0
        self.model_save_time = 0

    def run(self):

        start_time = time.time()
        last_time = start_time

        self.logger.info("Beginning training for {} timesteps".format(self.args.t_max))

        while self.runner.t_env <= self.args.t_max:

            # Run for a whole episode at a time
            episode_batch = self.runner.run(test_mode=False)
            self.buffer.insert_episode_batch(episode_batch)

            if self.buffer.can_sample(self.args.batch_size):
                episode_sample = self.buffer.sample(self.args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != self.args.device:
                    episode_sample.to(self.args.device)

                self.learner.train(episode_sample, self.runner.t_env, self.episode)

            # Execute test runs once in a while
            n_test_runs = max(1, self.args.test_nepisode // self.runner.batch_size)
            if (self.runner.t_env - self.last_test_T) / self.args.test_interval >= 1.0:

                self.logger.console_logger.info(
                    "t_env: {} / {}".format(self.runner.t_env, self.args.t_max)
                )
                self.logger.console_logger.info(
                    "Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, self.last_test_T, self.runner.t_env, self.args.t_max),
                        time_str(time.time() - start_time),
                    )
                )
                last_time = time.time()

                self.last_test_T = self.runner.t_env
                for _ in range(n_test_runs):
                    self.runner.run(test_mode=True)

            # Execute evaluate in target task once in a while
            n_eval_runs = max(1, self.args.eval_nepisode // self.runner.batch_size)
            if (self.runner.t_env - self.last_eval_T) / self.args.eval_interval >= 1.0:
                self.last_eval_T = self.runner.t_env
                for _ in range(n_eval_runs):
                    self.runner.run(test_mode=True, test_tag=True)

            if self.args.save_model and (
                    self.runner.t_env - self.model_save_time >= self.args.save_model_interval
                    or self.model_save_time == 0
            ):
                self.model_save_time = self.runner.t_env
                save_path = os.path.join(
                    self.args.local_results_path, "models", self.args.unique_token, str(self.runner.t_env)
                )
                # "results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                self.logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                self.learner.save_models(save_path)

            self.episode += self.args.batch_size_run

            if (self.runner.t_env - self.last_log_T) >= self.args.log_interval:
                self.logger.log_stat("episode", self.episode, self.runner.t_env)
                self.logger.print_recent_stats()
                self.last_log_T = self.runner.t_env

        self.runner.close_env()
        self.logger.console_logger.info("Finished Training")

    def evaluate_sequential(self):

        for _ in range(self.args.test_nepisode):
            self.runner.run(test_mode=True)

        if self.args.save_replay:
            self.runner.save_replay()

        self.runner.close_env()

    def eval_tag_task(self, n_episode=0):
        if not n_episode:
            n_episode = self.args.eval_nepisode
        stats = {}
        for _ in range(n_episode):
            self.runner.run(test_mode=True, test_tag=True, tag_eval_stats=stats)
        return stats
