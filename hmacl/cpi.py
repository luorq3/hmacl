from copy import deepcopy

import numpy as np


class CPI:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.update_type = args.update_type  # soft or hard
        self.metrics_key = args.metrics  # win_rate or return or ...
        self.agent = None
        self.stats = None  # {'returns': [], 'stats': {}}
        self.cur_metrics = None

    def update_stats(self, algo, mac):
        self.agent = deepcopy(mac.agent)
        self.stats, self.cur_metrics = self.eval_on_tag(algo)

    def improve(self, algo, mac):
        stats, metrics = self.eval_on_tag(algo)

        if self.update_type == "hard":
            assert self.cur_metrics is not None, "Please invoke `update_stats`"
            if metrics > self.cur_metrics:
                self.update_agent(mac, "hard", stats, metrics)
        elif self.update_type == "soft":
            self.update_agent(mac, "soft")
            self.update_stats(algo, mac)
        else:
            raise ValueError("No such model update approach: {}.".format(self.update_type))

    def update_agent(self, mac, update_type, stats=None, metrics=None):
        if update_type == "hard":
            self.agent = deepcopy(mac.agent)
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
