import torch.nn as nn
from hmacl.algo.modules.agents.rnn_agent import RNNAgent
import torch as th

class RNNCSAgent(nn.Module):
    def __init__(self, input_shapes, args):
        super(RNNCSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.cp2ap = args.cp2ap
        self.n_cs_agents = {k: len(v) for k, v in self.cp2ap.items()}
        self.input_shapes = input_shapes
        self.agents = th.nn.ModuleDict({k: RNNAgent(v, args) for k, v in self.input_shapes.items()})

    def init_hidden(self):
        # make hidden states on same device as model
        return {k: v.init_hidden() for k, v in self.agents.items()}

    def forward(self, inputs, hidden_states):
        hiddens = {}
        qs = []
        for k in self.agents.keys():
            input_ = inputs[k]
            hidden_state = hidden_states[k]
            q, h = self.agents[k](input_, hidden_state)
            hiddens[k] = h.unsqueeze(0)
            qs.append(q.unsqueeze(0))
        return th.cat(qs, dim=1), hiddens

    def cuda(self, device="cuda:0"):
        for _, a in self.agents.items():
            a.cuda(device=device)
