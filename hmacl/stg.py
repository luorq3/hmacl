import numpy as np


class TaskList:

    def __init__(self, env_name, scenario, **kwargs):
        self.env = env_name
        self.scenario = scenario

        self.initial_len = kwargs["initial_len"]
        self.tag_len = kwargs["tag_len"]
        self.expand_d = kwargs["expand_d"]
        self.tasks = {_id: [x, x] for _id, x in enumerate(
            np.arange(self.initial_len, self.tag_len, self.expand_d))}
        self.tasks[len(self.tasks)] = [self.tag_len] * 2

    def __len__(self):
        return len(self.tasks)


class STG:

    def __init__(self, env_name, scenario, **kwargs):
        self.tasks = TaskList(env_name, scenario, **kwargs)
        self.task_seen = np.zeros(len(self.tasks))
        self.max_task = 0
        self.ms = np.zeros(len(self.tasks))
        self.ls = np.zeros(len(self.tasks))
        self.loss_coef = kwargs['loss_coef']
        self.temperature = kwargs['temperature']

    def generate(self, task, metrics, loss):
        if not self.task_seen[task]:
            self.task_seen[task] = 1
            self.max_task = task

        self.ms[task] = np.abs(metrics)
        self.ls[task] = np.abs(loss)

        if np.all(self.task_seen) or np.random.random() < self.task_seen.sum() / len(self.task_seen):
            wms = self.replay_weight(self.ms)
            wls = self.replay_weight(self.ls)
            task_weights = (1 - self.loss_coef) * wms + self.loss_coef * wls
            if np.isclose(np.sum(task_weights), 0):
                task_weights = np.ones_like(task_weights, dtype=np.float) / len(task_weights)

            task_idx = np.random.choice(range(len(self.task_seen)), 1, p=task_weights)[0]
        else:
            self.max_task += 1
            task_idx = self.max_task

        return task_idx

    def replay_weight(self, scores):
        temp = np.flip(scores.argsort())
        ranks = np.zeros_like(temp)
        ranks[temp] = np.arange(len(temp)) + 1
        weights = 1 / ranks ** (1. / self.temperature) * self.task_seen

        z = np.sum(weights)
        if z > 0:
            weights /= z

        return weights


if __name__ == "__main__":
    np.random.seed(0)
    stg = STG("m2ale", "test", initial_len=5, tag_len=29, expand_d=5, temperature=1, loss_coef=0.5)
    task = 0
    ts = []
    for _ in range(10):
        ts.append(task)
        task = stg.generate(task, np.random.random(), np.random.random())
    print(ts)
