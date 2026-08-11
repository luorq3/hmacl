"""Environment-specific MATDD adapters."""

from .football import FootballTaskAdapter
from .m2ale import M2ALETaskAdapter
from .padded_env import PaddedMultiAgentEnv
from .pursuit import PursuitTaskAdapter

__all__ = [
    "FootballTaskAdapter",
    "M2ALETaskAdapter",
    "PaddedMultiAgentEnv",
    "PursuitTaskAdapter",
]
