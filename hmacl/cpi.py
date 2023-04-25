
class CPI:

    def __init__(self, tag_task, model, ps_type, update_type, ap2cp=None, **kwargs):
        self.env = tag_task
        self.model = model
        self.ap2cp = ap2cp
        self.ps_type = ps_type
        self.update_type = update_type
        self.stats = self.eval(self.model)
        self.eval_episode = kwargs['eval_episode']
        # m--e.g."win-rate", "return".the metric compare new model and current model
        self.m = kwargs['m']

    def improve(self, new_model):
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

    def eval(self, new_model):
        self.env.clean_stats()
        for episode in range(self.eval_episode):
            self.env.reset()
            done = False
            while not done:
                obs = self.env.get_obs()
                action = self.act(obs, new_model)
                reward, done, info = self.env.step(action)
        return self.env.get_stats()

    def update_model(self, new_model):
        pass

    def act(self, obs, model):
        action = []
        if self.ps_type == "ps":
            action = model.act(obs)
        else:
            if self.ps_type == "no_ps":
                assert len(model) == len(obs), "Model number can't match obs number when disable parameter sharing."
            elif self.ps_type == "class_ps":
                assert len(model) <= len(obs)

            for i, o in enumerate(obs):
                m = self.ap2cp(i, model)
                a = m.act(o)
                action.append(a)

        assert len(action) == len(obs), "Action number can't match obs number."

        return action



