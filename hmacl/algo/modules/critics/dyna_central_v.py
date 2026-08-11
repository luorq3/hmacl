"""MAPPO critic with DyNA-style other-agent aggregation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmacl.algo.modules.dyna import DynaObservationEncoder


class DynaCentralVCritic(nn.Module):
    """Predict per-agent values from own and aggregated teammate embeddings."""

    def __init__(self, scheme, args):
        super().__init__()
        self.n_agents = args.n_agents
        self.output_type = "v"
        hidden_dim = int(getattr(args, "dyna_hidden_dim", 128))
        embedding_dim = int(getattr(args, "dyna_embedding_dim", 64))
        self.encoder = DynaObservationEncoder(
            scheme["obs"]["vshape"], hidden_dim, embedding_dim
        )
        self.fc1 = nn.Linear(embedding_dim * 2 + self.n_agents, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.output = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        time_slice = slice(None) if t is None else slice(t, t + 1)
        observations = batch["obs"][:, time_slice]
        embeddings = self.encoder(observations)
        if "agent_mask" in batch.scheme:
            agent_mask = batch["agent_mask"][:, time_slice].to(embeddings.dtype)
        else:
            agent_mask = torch.ones(
                *embeddings.shape[:-1],
                1,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
        active_embeddings = embeddings * agent_mask
        team_sum = active_embeddings.sum(dim=2, keepdim=True)
        other_sum = team_sum - active_embeddings
        other_count = (agent_mask.sum(dim=2, keepdim=True) - agent_mask).clamp_min(1.0)
        other_mean = other_sum / other_count
        batch_size, sequence_length = observations.shape[:2]
        agent_ids = (
            torch.eye(self.n_agents, device=batch.device)
            .view(1, 1, self.n_agents, self.n_agents)
            .expand(batch_size, sequence_length, -1, -1)
        )
        features = torch.cat([active_embeddings, other_mean, agent_ids], dim=-1)
        hidden = F.relu(self.fc1(features))
        hidden = F.relu(self.fc2(hidden))
        return self.output(hidden)
