
class CPI:

    def __init__(self, tag_task, model, ps_type, ap2cp=None, **kwargs):
        self.env = tag_task
        self.model = model
        self.ap2cp = ap2cp
        self.ps_type = ps_type
        self.eval(self.model)
        self.eval_episode = kwargs['eval_episode']

    def improve(self, new_model):
        pass

    def eval(self, new_model):
        self.env.clean_stats()
        for episode in range(self.eval_episode):
            self.env.reset()
            done = False
            while not done:
                obs = self.env.get_obs()
                action = self.act(obs, new_model)
                reward, done, info = self.env.step(action)
        stats = self.env.get_stats()


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



