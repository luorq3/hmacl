"""Map-size curriculum adapter for the existing M2ALE environment."""

from copy import deepcopy
from typing import Mapping

from ..task_space import Number, ParameterSpec, TaskParameterSpace


class M2ALETaskAdapter:
    """Expose square M2ALE map size as a one-dimensional task family.

    M2ALE currently keeps the agent population fixed.  This adapter is useful
    for integration tests and map-size experiments, but it does not reproduce
    the paper's variable-agent experiments.
    """

    parameter_name = "map_size"

    def __init__(self, base_config: Mapping[str, object]):
        self.base_config = deepcopy(dict(base_config))
        initial_size = int(self.base_config["map_size"][0])
        target_size = int(self.base_config["tag_map_size"][0])
        self.space = TaskParameterSpace(
            [
                ParameterSpec(
                    self.parameter_name, initial_size, target_size, integer=True
                )
            ]
        )
        self.initial_task = {self.parameter_name: initial_size}
        self.target_task = {self.parameter_name: target_size}

    def env_config(self, task: Mapping[str, Number]):
        config = deepcopy(self.base_config)
        size = int(task[self.parameter_name])
        config["map_size"] = [size, size]
        return config
