import copy

from hmacl.algo.modules.agents import RNNCSAgent
from hmacl.algo.components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

class ClassSharedMAC:
    def __init__(self, scheme, groups, args):
        """
        ap2cp: ['0' '0' '1' '1' '1' '3' '3']
        cp2ap: {'0': [0, 1], '1': [2, 3, 4], '3': [5, 6]}
        """
        self.n_agents = args.n_agents
        # cpi
        assert args.cps, "Class policy sharing should toggle `cps` to True."
        assert args.agent == "rnn_cs"
        self.ap2cp = args.ap2cp
        self.cp2ap = {}
        for i_agent, cls in enumerate(self.ap2cp):
            if self.cp2ap.__contains__(cls):
                self.cp2ap.get(cls).append(i_agent)
            else:
                self.cp2ap[cls] = [i_agent]
        args.cp2ap = self.cp2ap
        self.args = args

        input_shapes = {}
        for k, v in self.cp2ap.items():
            input_shapes[k] = self._get_input_shape(scheme, len(v))
        self._build_agents(input_shapes)

        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = {k: v.unsqueeze(0).expand(batch_size, len(self.cp2ap[k]), -1) for k, v in self.agent.init_hidden().items()}

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def soft_update_model(self, path, tau):
        source_agent = copy.deepcopy(self.agent)
        source_agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        for target_param, param in zip(self.agent.parameters(), source_agent.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)

    def _build_agents(self, input_shapes):
        self.agent = RNNCSAgent(input_shapes, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = {}
        obs = batch["obs"][:, t]
        for k in self.cp2ap.keys():
            input_ = [obs[:, self.cp2ap[k]]]
            if self.args.obs_last_action:
                if t == 0:
                    input_.append(th.zeros_like(batch["actions_onehot"][:, t, self.cp2ap[k]]))
                else:
                    input_.append(batch["actions_onehot"][:, t - 1, self.cp2ap[k]])
            if self.args.obs_agent_id:
                input_.append(th.eye(len(self.cp2ap[k]), device=batch.device).unsqueeze(0).expand(bs, -1, -1))

            input_ = th.cat([x.reshape(bs * len(self.cp2ap[k]), -1) for x in input_], dim=1)
            inputs[k] = input_
        return inputs

    def _get_input_shape(self, scheme, n_ca):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += n_ca
        return input_shape
