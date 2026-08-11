from functools import partial

from hmacl.algo.envs.pursuit import PettingZooPursuitEnv
from hmacl.algo.envs.vmas import VMASFootballEnv
from hmacl.envs.m2ale.m2ale_agent_vs_bot import M2ALE
from hmacl.envs.m2ale.multiagentenv import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "m2ale": partial(env_fn, env=M2ALE),
    "pursuit": partial(env_fn, env=PettingZooPursuitEnv),
    "vmas": partial(env_fn, env=VMASFootballEnv),
}
