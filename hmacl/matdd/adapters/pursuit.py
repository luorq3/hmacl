"""Map-size and pursuer-count curriculum adapter for SISL Pursuit."""

from copy import deepcopy
from typing import Mapping

from ..task_space import Number, ParameterSpec, TaskParameterSpace


class PursuitTaskAdapter:
    """Expose the two free parameters plotted in the MATDD manuscript."""

    def __init__(self, base_config: Mapping[str, object]):
        self.base_config = deepcopy(dict(base_config))
        initial_map_size = int(self.base_config["x_size"])
        if int(self.base_config["y_size"]) != initial_map_size:
            raise ValueError("Pursuit curricula currently require square maps")
        initial_agents = int(self.base_config["n_pursuers"])
        target_map_size = int(self.base_config.pop("target_map_size"))
        target_agents = int(self.base_config.pop("target_n_pursuers"))
        self.scale_n_evaders = bool(self.base_config.pop("scale_n_evaders", False))
        self.initial_evaders = int(self.base_config["n_evaders"])
        self.space = TaskParameterSpace(
            [
                ParameterSpec(
                    "map_size", initial_map_size, target_map_size, integer=True
                ),
                ParameterSpec(
                    "n_pursuers", initial_agents, target_agents, integer=True
                ),
            ]
        )
        self.initial_task = {
            "map_size": initial_map_size,
            "n_pursuers": initial_agents,
        }
        self.target_task = {
            "map_size": target_map_size,
            "n_pursuers": target_agents,
        }

    def env_config(self, task: Mapping[str, Number]):
        config = deepcopy(self.base_config)
        map_size = int(task["map_size"])
        pursuers = int(task["n_pursuers"])
        config["x_size"] = map_size
        config["y_size"] = map_size
        config["n_pursuers"] = pursuers
        if self.scale_n_evaders:
            ratio = pursuers / float(self.initial_task["n_pursuers"])
            config["n_evaders"] = max(1, round(self.initial_evaders * ratio))
        return config
