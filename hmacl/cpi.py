from copy import deepcopy

import numpy as np


class CPI:

    def __init__(self, args, cur_stats, mac):
        self.update_type = args.update_type
        self.stats = cur_stats  # {'returns': [], 'stats': {'battle_won': 0, 'episode_limit': 1, 'n_episodes': 1, 'ep_length': 200}}
        self.mac = deepcopy(mac)
        self.metrics = "win_rate"  # args.metrics = win_rate or return

    def cal_metrics(self):
        if self.metrics == "win_rate":
            battle_won = self.stats["battle_won"]
            n_episodes = self.stats["n_episodes"]
            metrics = float(battle_won) / n_episodes
        elif self.metrics == "return":
            returns = self.stats["returns"]
            metrics = np.mean(returns)
        else:
            raise NotImplementedError("No such metrics: {}".format(self.metrics))
        return metrics

    def improve(self, new_mac):
        new_stats = self.eval(new_model)

        # update_model
        if self.update_type == "soft":
            self.update_model(new_model)
        elif self.update_type == "hard":
            if new_stats[self.m] >= self.stats[self.m]:
                self.update_model(new_model)
        else:
            raise ValueError(f"No such model update type: {self.update_type}")

        return self.model, new_stats
