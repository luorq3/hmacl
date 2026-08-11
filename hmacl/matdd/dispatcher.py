"""Historical curriculum replay for MATDD."""

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .task_space import Number, Task, TaskParameterSpace


@dataclass
class CurriculumRecord:
    task: Task
    training_value: float
    target_distance: float
    visits: int = 1
    insertion_index: int = 0


class CurriculumDispatcher:
    """Prioritize tasks by learning value and proximity to the target.

    The replay distribution follows the paper:

        P(replay task) = (1 - rho) * P(training value)
                         + rho * P(scale rank)

    Training-value ranks are descending; target-distance ranks are ascending.
    Replaying a task never removes it from the pool.  When the pool is full,
    FIFO eviction is used because the original manuscript does not specify an
    eviction rule.
    """

    def __init__(
        self,
        space: TaskParameterSpace,
        target_task: Mapping[str, Number],
        capacity: int,
        rho: float = 0.5,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ):
        if capacity <= 0:
            raise ValueError("dispatcher capacity must be positive")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.space = space
        self.target_task = dict(target_task)
        self.capacity = capacity
        self.rho = rho
        self.temperature = temperature
        self._records: "OrderedDict[Tuple[float, ...], CurriculumRecord]" = (
            OrderedDict()
        )
        self._rng = random.Random(seed)
        self._insertion_count = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def fill_ratio(self) -> float:
        return len(self) / float(self.capacity)

    @property
    def records(self) -> Tuple[CurriculumRecord, ...]:
        return tuple(self._records.values())

    def observe(self, task: Mapping[str, Number], training_value: float) -> None:
        if not math.isfinite(training_value):
            raise ValueError("training value must be finite")
        task_copy = dict(task)
        key = self.space.key(task_copy)
        value = abs(float(training_value))
        if key in self._records:
            record = self._records[key]
            record.task = task_copy
            record.training_value = value
            record.target_distance = self.space.distance(task_copy, self.target_task)
            record.visits += 1
            return

        if len(self._records) >= self.capacity:
            self._records.popitem(last=False)
        self._insertion_count += 1
        self._records[key] = CurriculumRecord(
            task=task_copy,
            training_value=value,
            target_distance=self.space.distance(task_copy, self.target_task),
            insertion_index=self._insertion_count,
        )

    def probabilities(self) -> Tuple[float, ...]:
        records = self.records
        if not records:
            return tuple()
        value_distribution = self._rank_distribution(
            [record.training_value for record in records], descending=True
        )
        scale_distribution = self._rank_distribution(
            [record.target_distance for record in records], descending=False
        )
        probabilities = [
            (1.0 - self.rho) * value_probability + self.rho * scale_probability
            for value_probability, scale_probability in zip(
                value_distribution, scale_distribution
            )
        ]
        total = sum(probabilities)
        return tuple(probability / total for probability in probabilities)

    def sample(self) -> Task:
        records = self.records
        if not records:
            raise RuntimeError("cannot replay from an empty curriculum pool")
        index = self._rng.choices(
            range(len(records)), weights=self.probabilities(), k=1
        )[0]
        return dict(records[index].task)

    def snapshot(self) -> List[Dict[str, object]]:
        probabilities = self.probabilities()
        return [
            {
                "task": dict(record.task),
                "training_value": record.training_value,
                "target_distance": record.target_distance,
                "visits": record.visits,
                "probability": probabilities[index],
            }
            for index, record in enumerate(self.records)
        ]

    def _rank_distribution(
        self, scores: Sequence[float], descending: bool
    ) -> Tuple[float, ...]:
        ordered_scores = sorted(set(scores), reverse=descending)
        rank_by_score = {
            score: rank for rank, score in enumerate(ordered_scores, start=1)
        }
        ranks = [rank_by_score[score] for score in scores]
        exponent = 1.0 / self.temperature
        weights = [1.0 / (rank**exponent) for rank in ranks]
        total = sum(weights)
        return tuple(weight / total for weight in weights)
