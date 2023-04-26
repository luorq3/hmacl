
class STG:

    def __init__(self, n_curricula=None):
        self.task_seen = {}
        self.n_task = n_curricula
        self.score = {}
        self.pd = None
        self.ps = None

    def generate(self, task, loss, metrics):
        if task.id not in self.task_seen.keys():
            self.task_seen[task.id] = task

