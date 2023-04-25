

class Algo:

    def __init__(self, model, task, ap2cp, n_timestep):
        self.model = model
        self.env = task
        self.ap2cp = ap2cp
        self.n_timestep = n_timestep

    def train(self):
        timestep = 0
        while timestep < self.n_timestep:
            pass