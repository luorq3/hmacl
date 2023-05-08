from copy import deepcopy

import numpy as np


class TaskList:

    def __init__(self, args):
        self.env = args.env
        self.task_args = args.env_args
        initial_map_x = self.task_args['map_size'][0]
        tag_map_x = self.task_args['tag_map_size'][0]
        expand_d = self.task_args['expand_d']
        self.tasks = {_id: [x, x] for _id, x in enumerate(
            np.arange(initial_map_x, tag_map_x, expand_d))}
        self.tag_task_id = len(self.tasks)
        self.tasks[self.tag_task_id] = [tag_map_x] * 2

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):
        task_args = deepcopy(self.task_args)
        task_args['map_size'] = self.tasks[item]
        return task_args


class STG:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.tasks = TaskList(args)
        self.task_seen = np.zeros(len(self.tasks))
        self.max_task = 0
        self.ms = np.zeros(len(self.tasks))
        self.ls = np.zeros(len(self.tasks))
        self.loss_coef = self.args.loss_coef
        self.temperature = self.args.temperature
        self.tag_task_id = self.tasks.tag_task_id

    def generate(self, task, metrics, td_error):
        if not self.task_seen[task]:
            self.task_seen[task] = 1
            self.max_task = task

        self.ms[task] = np.abs(metrics)
        self.ls[task] = np.abs(td_error)

        if np.all(self.task_seen) or np.random.random() < max(self.task_seen.sum() / len(self.task_seen), 0.5):
            wms = self.replay_weight(self.ms)
            wls = self.replay_weight(self.ls)
            task_weights = (1 - self.loss_coef) * wms + self.loss_coef * wls
            if np.isclose(np.sum(task_weights), 0):
                task_weights = np.ones_like(task_weights, dtype=np.float) / len(task_weights)
            cur_task_id = np.random.choice(range(len(self.task_seen)), 1, p=task_weights)[0]
        else:
            self.max_task += 1
            cur_task_id = self.max_task
        return cur_task_id

    def get_task_config(self, task_id):
        return self.tasks[task_id]

    def get_tag_task_config(self):
        return self.get_task_config(self.tag_task_id)

    def replay_weight(self, scores):
        temp = np.flip(scores.argsort())
        ranks = np.zeros_like(temp)
        ranks[temp] = np.arange(len(temp)) + 1
        weights = 1 / ranks ** (1. / self.temperature) * self.task_seen

        z = np.sum(weights)
        if z > 0:
            weights /= z

        return weights


# if __name__ == "__main__":
#     np.random.seed(0)
#     stg = STG("m2ale", "test", initial_len=5, tag_len=29, expand_d=5, temperature=1, loss_coef=0.5)
#     task = 0
#     ts = []
#     for _ in range(10):
#         ts.append(task)
#         stg.generate(task, np.random.random(), np.random.random())
#     print(ts)
