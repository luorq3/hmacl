"""QMIX conditioned on a DyNA-style variable-team embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmacl.algo.modules.dyna import DynaObservationEncoder, masked_agent_mean


class DynaQMixer(nn.Module):
    """Use masked observation aggregation instead of a padded global state."""

    uses_observations = True

    def __init__(self, args):
        super().__init__()
        self.n_agents = args.n_agents
        self.mixing_dim = args.mixing_embed_dim
        dyna_hidden_dim = int(getattr(args, "dyna_hidden_dim", 128))
        self.dyna_embedding_dim = int(getattr(args, "dyna_embedding_dim", 64))
        self.encoder = DynaObservationEncoder(
            args.obs_shape, dyna_hidden_dim, self.dyna_embedding_dim
        )
        hypernet_embed = int(getattr(args, "hypernet_embed", 64))
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.dyna_embedding_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.mixing_dim * self.n_agents),
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.dyna_embedding_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.mixing_dim),
        )
        self.hyper_b_1 = nn.Linear(self.dyna_embedding_dim, self.mixing_dim)
        self.value = nn.Sequential(
            nn.Linear(self.dyna_embedding_dim, self.mixing_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_dim, 1),
        )

    def forward(self, agent_qs, states=None, observations=None, agent_mask=None):
        if observations is None:
            raise ValueError("DynaQMixer requires per-agent observations")
        batch_size = agent_qs.size(0)
        embeddings = self.encoder(observations)
        team_embedding = masked_agent_mean(embeddings, agent_mask)
        flat_team = team_embedding.reshape(-1, self.dyna_embedding_dim)

        if agent_mask is not None:
            agent_qs = agent_qs * agent_mask.to(agent_qs.dtype)
        flat_qs = agent_qs.reshape(-1, 1, self.n_agents)
        first_weights = torch.abs(self.hyper_w_1(flat_team)).reshape(
            -1, self.n_agents, self.mixing_dim
        )
        first_bias = self.hyper_b_1(flat_team).reshape(-1, 1, self.mixing_dim)
        hidden = F.elu(torch.bmm(flat_qs, first_weights) + first_bias)
        final_weights = torch.abs(self.hyper_w_final(flat_team)).reshape(
            -1, self.mixing_dim, 1
        )
        value = self.value(flat_team).reshape(-1, 1, 1)
        mixed = torch.bmm(hidden, final_weights) + value
        return mixed.reshape(batch_size, -1, 1)
