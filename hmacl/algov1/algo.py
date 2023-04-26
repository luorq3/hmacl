from hmacl.algov1.episode_runner import EpisodeRunner

class Algo:

    def __init__(self, policies, task, tag_task, ap2cp, n_timestep):
        self.policies = policies
        self.env = task
        self.eval_env = tag_task
        self.n_timestep = n_timestep

        # TODO
        self.ap2cp = ap2cp
        agents_policies = self.env.get_agents_types
        num_agents = len(agents_policies)
        policies_agents = None
        hidden_size = None
        self.runner = EpisodeRunner(self.env, self.eval_env, self.policies, agents_policies, policies_agents, num_agents, hidden_size)

    def train(self):
        timestep = 0
        while timestep < self.n_timestep:
            pass