import os
from copy import deepcopy

import numpy as np


class CPI:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.update_type = args.update_type  # soft or hard
        self.metrics_key = args.metrics  # win_rate or return or ...
        self.agent_path = None
        self.stats = None  # {'returns': [], 'stats': {}}
        self.cur_metrics = None

    def update_stats(self, algo, learner, t_env):
        self.save_model(learner, t_env)
        self.stats, self.cur_metrics = self.eval_on_tag(algo)

    def improve(self, algo, learner, t_env):
        stats, metrics = self.eval_on_tag(algo)

        if self.update_type == "hard":
            assert self.cur_metrics is not None, "Please invoke `update_stats` before `improve`."
            if metrics > self.cur_metrics:
                self.update_agent(learner, "hard", t_env, stats, metrics)
        elif self.update_type == "soft":
            self.update_agent(learner, "soft", t_env)
            self.update_stats(algo, learner, t_env)
        else:
            raise ValueError("No such model update approach: {}.".format(self.update_type))

    def update_agent(self, learner, update_type, t_env, stats=None, metrics=None):
        if update_type == "hard":
            self.save_model(learner, t_env)
            self.stats = stats
            self.cur_metrics = metrics
        else:
            # todo copy parameters proportionally to self.agent, no need update mac.agent
            # 1. 将旧模型的一部份复制到新模型
            # 2. 调用update_stats更新self.agent和metrics
            pass

    def eval_on_tag(self, algo):
        stats = algo.eval_tag_task()
        metrics = self.cal_metrics(stats)
        self.logger.console_logger.info("Evaluate target task get metrics: {}".format(metrics))
        return stats, metrics

    def cal_metrics(self, stats):
        if self.metrics_key == "win_rate":
            battle_won = stats["battle_won"]
            n_episodes = stats["n_episodes"]
            metrics = float(battle_won) / n_episodes
        elif self.metrics_key == "return":
            returns = stats["returns"]
            metrics = np.mean(returns)
        else:
            raise NotImplementedError("No such metrics: {}".format(self.metrics_key))
        return metrics

    def save_model(self, learner, t_env):
        save_path = os.path.join(
            self.args.local_results_path, "models", self.args.unique_token, "cpi", str(t_env)
        )
        os.makedirs(save_path, exist_ok=True)
        self.logger.console_logger.info("CPI saving models to {} at timestep {}".format(save_path, t_env))
        learner.save_models(save_path)
        self.agent_path = save_path
