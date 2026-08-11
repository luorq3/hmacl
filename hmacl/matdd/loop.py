"""Environment-independent MATDD training state machine."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Protocol

from .designer import DesignerAction, DesignerContext
from .dispatcher import CurriculumDispatcher
from .task_space import Number, Task


@dataclass(frozen=True)
class TrainingResult:
    mean_return: float
    mean_abs_td_error: float
    environment_steps: int
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    mean_return: float
    environment_steps: int = 0
    metrics: Mapping[str, float] = field(default_factory=dict)


class StudentBackend(Protocol):
    """Bridge between MATDD and a concrete MARL implementation."""

    def train(self, task: Mapping[str, Number], step_budget: int) -> TrainingResult: ...

    def evaluate(
        self, task: Mapping[str, Number], episodes: int
    ) -> EvaluationResult: ...


class CurriculumDesigner(Protocol):
    @property
    def pending_transitions(self) -> int: ...

    def propose(self, context: DesignerContext) -> DesignerAction: ...

    def observe(
        self, action: DesignerAction, reward: float, done: bool = False
    ) -> None: ...

    def update(
        self, last_context: Optional[DesignerContext] = None
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class MATDDConfig:
    curriculum_iterations: int
    steps_per_curriculum: int
    final_target_steps: int
    evaluation_episodes: int = 32
    teacher_update_interval: int = 8
    seed: Optional[int] = None

    def __post_init__(self):
        if self.curriculum_iterations <= 0:
            raise ValueError("curriculum_iterations must be positive")
        if self.steps_per_curriculum <= 0 or self.final_target_steps < 0:
            raise ValueError("training step budgets are invalid")
        if self.evaluation_episodes <= 0 or self.teacher_update_interval <= 0:
            raise ValueError("evaluation and teacher intervals must be positive")


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    source: str
    task: Task
    protagonist_return: float
    antagonist_return: float
    parallel_team_regret: float
    target_return: float
    training_value: float
    environment_steps: int
    teacher_metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MATDDRunResult:
    iterations: List[IterationResult]
    final_training: Optional[TrainingResult]
    total_environment_steps: int


class MATDDTrainingLoop:
    """Train protagonist and antagonist on designed or replayed tasks.

    The antagonist and protagonist receive the same task and interaction
    budget.  The designer reward is ``antagonist return - protagonist return``.
    Evaluation steps are included in total environment-step accounting.
    """

    def __init__(
        self,
        protagonist: StudentBackend,
        antagonist: StudentBackend,
        designer: CurriculumDesigner,
        dispatcher: CurriculumDispatcher,
        initial_task: Mapping[str, Number],
        target_task: Mapping[str, Number],
        config: MATDDConfig,
    ):
        self.protagonist = protagonist
        self.antagonist = antagonist
        self.designer = designer
        self.dispatcher = dispatcher
        self.initial_task = dict(initial_task)
        self.target_task = dict(target_task)
        self.config = config
        self._rng = random.Random(config.seed)

    def run(self) -> MATDDRunResult:
        history: List[IterationResult] = []
        current_task = dict(self.initial_task)
        target_return = self.protagonist.evaluate(
            self.target_task, self.config.evaluation_episodes
        )
        total_environment_steps = target_return.environment_steps

        for iteration in range(self.config.curriculum_iterations):
            context = self._context(iteration, current_task, target_return.mean_return)
            replay = (
                len(self.dispatcher) > 0
                and self._rng.random() < self.dispatcher.fill_ratio
            )
            action = None
            if replay:
                source = "dispatcher"
                task = self.dispatcher.sample()
            else:
                source = "designer"
                action = self.designer.propose(context)
                task = dict(action.task)

            protagonist_result = self.protagonist.train(
                task, self.config.steps_per_curriculum
            )
            antagonist_result = self.antagonist.train(
                task, self.config.steps_per_curriculum
            )
            target_evaluation = self.protagonist.evaluate(
                self.target_task, self.config.evaluation_episodes
            )
            regret = antagonist_result.mean_return - protagonist_result.mean_return
            self.dispatcher.observe(task, protagonist_result.mean_abs_td_error)
            teacher_metrics: Dict[str, float] = {}
            if action is not None:
                self.designer.observe(action, regret, done=False)

            next_context = self._context(
                iteration + 1, task, target_evaluation.mean_return
            )
            if self.designer.pending_transitions >= self.config.teacher_update_interval:
                teacher_metrics.update(self.designer.update(next_context))

            iteration_steps = (
                protagonist_result.environment_steps
                + antagonist_result.environment_steps
                + target_evaluation.environment_steps
            )
            total_environment_steps += iteration_steps
            history.append(
                IterationResult(
                    iteration=iteration,
                    source=source,
                    task=dict(task),
                    protagonist_return=protagonist_result.mean_return,
                    antagonist_return=antagonist_result.mean_return,
                    parallel_team_regret=regret,
                    target_return=target_evaluation.mean_return,
                    training_value=protagonist_result.mean_abs_td_error,
                    environment_steps=iteration_steps,
                    teacher_metrics=teacher_metrics,
                )
            )
            current_task = dict(task)
            target_return = target_evaluation

        if self.designer.pending_transitions:
            final_context = self._context(
                self.config.curriculum_iterations,
                current_task,
                target_return.mean_return,
            )
            self.designer.update(final_context)

        final_training = None
        if self.config.final_target_steps:
            final_training = self.protagonist.train(
                self.target_task, self.config.final_target_steps
            )
            total_environment_steps += final_training.environment_steps
        return MATDDRunResult(
            iterations=history,
            final_training=final_training,
            total_environment_steps=total_environment_steps,
        )

    def _context(
        self, iteration: int, current_task: Mapping[str, Number], target_return: float
    ) -> DesignerContext:
        progress = min(1.0, iteration / float(self.config.curriculum_iterations))
        return DesignerContext(
            current_task=dict(current_task),
            target_task=dict(self.target_task),
            progress=progress,
            target_return=target_return,
            pool_fill_ratio=self.dispatcher.fill_ratio,
        )
