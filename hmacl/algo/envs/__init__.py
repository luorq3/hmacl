from functools import partial
from hmacl.envs.m2ale.m2ale_agent_vs_bot import CombatEnv
from hmacl.envs.m2ale.multiagentenv import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {"m2ale": partial(env_fn, env=CombatEnv)}
