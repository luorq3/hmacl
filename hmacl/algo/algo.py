import os
import time
from hmacl.algo.utils.timehelper import time_left, time_str


class Algo:

    def __init__(self, args, logger, runner, mac, learner, buffer):
        self.args = args
        self.logger = logger
        self.runner = runner
        self.mac = mac
        self.learner = learner
        self.buffer = buffer

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

        # count every run within t_update
        t_run = 0

        td_error = 0
        start_time = time.time()
        last_time = start_time

        self.logger.info("Beginning training for {} timesteps".format(self.args.t_update))

        while t_run <= self.args.t_update:

            # Run for a whole episode at a time
            episode_batch = self.runner.run(test_mode=False)
            t_run += self.runner.t

            self.buffer.insert_episode_batch(episode_batch)

            if self.buffer.can_sample(self.args.batch_size):
                episode_sample = self.buffer.sample(self.args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != self.args.device:
                    episode_sample.to(self.args.device)

                td_error = self.learner.train(episode_sample, self.runner.t_env, self.episode)

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
                    self.args.local_results_path, self.args.unique_token, "models", str(self.runner.t_env)
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
        return td_error

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
