from functools import partial
from hmacl.envs.beachlanding.combat_env_vs_bot import CombatEnv
from hmacl.envs.beachlanding.multiagentenv import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {"m2ale": partial(env_fn, env=CombatEnv)}
