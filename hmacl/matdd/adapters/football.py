"""Agent-count curriculum adapter for VMAS Football."""

from copy import deepcopy
from typing import Mapping

from ..task_space import Number, ParameterSpec, TaskParameterSpace


class FootballTaskAdapter:
    """Scale the controlled blue team, and optionally the opposing red team."""

    parameter_name = "n_agents"

    def __init__(self, base_config: Mapping[str, object]):
        self.base_config = deepcopy(dict(base_config))
        if self.base_config.get("scenario") != "football":
            raise ValueError("FootballTaskAdapter requires scenario='football'")
        initial_agents = int(self.base_config["n_blue_agents"])
        target_agents = int(self.base_config.pop("target_n_agents"))
        if target_agents < initial_agents:
            raise ValueError("target_n_agents must not be smaller than n_blue_agents")
        self.scale_red_team = bool(self.base_config.pop("scale_red_team", True))
        self.space = TaskParameterSpace(
            [
                ParameterSpec(
                    self.parameter_name,
                    initial_agents,
                    target_agents,
                    integer=True,
                )
            ]
        )
        self.initial_task = {self.parameter_name: initial_agents}
        self.target_task = {self.parameter_name: target_agents}

    def env_config(self, task: Mapping[str, Number]):
        config = deepcopy(self.base_config)
        agents = int(task[self.parameter_name])
        config["n_blue_agents"] = agents
        if self.scale_red_team:
            config["n_red_agents"] = agents
        return config
