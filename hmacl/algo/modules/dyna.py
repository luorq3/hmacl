"""DyNA-style shared feature extraction for variable-size teams."""

import math

import torch
import torch.nn as nn


class DynaObservationEncoder(nn.Module):
    """Encode every agent observation with one shared network."""

    def __init__(self, obs_shape, hidden_dim, embedding_dim):
        super().__init__()
        self.obs_dim = (
            math.prod(obs_shape) if not isinstance(obs_shape, int) else obs_shape
        )
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        flattened = observations.reshape(*observations.shape[:3], self.obs_dim)
        return self.network(flattened)


def masked_agent_mean(embeddings, agent_mask=None):
    """Aggregate agents without allowing padded slots to change the result."""

    if agent_mask is None:
        agent_mask = torch.ones(
            *embeddings.shape[:-1],
            1,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
    elif agent_mask.ndim == embeddings.ndim - 1:
        agent_mask = agent_mask.unsqueeze(-1)
    agent_mask = agent_mask.to(dtype=embeddings.dtype)
    total = (embeddings * agent_mask).sum(dim=2)
    count = agent_mask.sum(dim=2).clamp_min(1.0)
    return total / count
