"""Task parameter normalization for MATDD curricula."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

Number = Union[int, float]
Task = Dict[str, Number]


@dataclass(frozen=True)
class ParameterSpec:
    """One bounded free parameter of a task family."""

    name: str
    minimum: float
    maximum: float
    integer: bool = False

    def __post_init__(self):
        if not self.name:
            raise ValueError("parameter name cannot be empty")
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("parameter bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("parameter maximum must be greater than minimum")


class TaskParameterSpace:
    """Maps environment parameters to and from the unit hypercube.

    Ranking and distances are computed after normalization, so parameters with
    different physical units contribute on the same scale.
    """

    def __init__(self, specs: Iterable[ParameterSpec]):
        self.specs: Tuple[ParameterSpec, ...] = tuple(specs)
        if not self.specs:
            raise ValueError("at least one task parameter is required")
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("task parameter names must be unique")

    @property
    def dimension(self) -> int:
        return len(self.specs)

    def normalize(self, task: Mapping[str, Number]) -> Tuple[float, ...]:
        values = []
        for spec in self.specs:
            if spec.name not in task:
                raise KeyError("task is missing parameter {!r}".format(spec.name))
            value = float(task[spec.name])
            if value < spec.minimum or value > spec.maximum:
                raise ValueError(
                    "parameter {!r}={} is outside [{}, {}]".format(
                        spec.name, value, spec.minimum, spec.maximum
                    )
                )
            values.append((value - spec.minimum) / (spec.maximum - spec.minimum))
        return tuple(values)

    def denormalize(self, vector: Sequence[Number]) -> Task:
        if len(vector) != self.dimension:
            raise ValueError(
                "expected {} normalized values, got {}".format(
                    self.dimension, len(vector)
                )
            )
        task: Task = {}
        for spec, raw_value in zip(self.specs, vector):
            normalized = min(1.0, max(0.0, float(raw_value)))
            value = spec.minimum + normalized * (spec.maximum - spec.minimum)
            task[spec.name] = int(round(value)) if spec.integer else value
        return task

    def clip_to_target(
        self, task: Mapping[str, Number], target: Mapping[str, Number]
    ) -> Task:
        """Clamp every curriculum dimension to the configured target task."""

        result: Task = {}
        for spec in self.specs:
            value = min(float(task[spec.name]), float(target[spec.name]))
            value = max(spec.minimum, value)
            result[spec.name] = int(round(value)) if spec.integer else value
        return result

    def distance(
        self, left: Mapping[str, Number], right: Mapping[str, Number]
    ) -> float:
        left_vector = self.normalize(left)
        right_vector = self.normalize(right)
        return math.sqrt(
            sum(
                (left_value - right_value) ** 2
                for left_value, right_value in zip(left_vector, right_vector)
            )
        )

    def key(self, task: Mapping[str, Number]) -> Tuple[float, ...]:
        """Return a stable, normalized identity for pool de-duplication."""

        return tuple(round(value, 12) for value in self.normalize(task))
