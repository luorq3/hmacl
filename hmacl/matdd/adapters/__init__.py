"""Environment-specific MATDD adapters."""

from .m2ale import M2ALETaskAdapter
from .padded_env import PaddedMultiAgentEnv

__all__ = ["M2ALETaskAdapter", "PaddedMultiAgentEnv"]
