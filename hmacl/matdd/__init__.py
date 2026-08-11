"""Core components for Multi-agent Automatic Task Design and Dispatch.

The package deliberately keeps the curriculum state machine independent from
the MARL implementation.  Environment- and learner-specific code should
implement :class:`hmacl.matdd.loop.StudentBackend`.
"""

from .designer import (
    DesignerAction,
    DesignerContext,
    PPOCurriculumDesigner,
    RandomCurriculumDesigner,
)
from .dispatcher import CurriculumDispatcher, CurriculumRecord
from .loop import (
    EvaluationResult,
    IterationResult,
    MATDDConfig,
    MATDDRunResult,
    MATDDTrainingLoop,
    TrainingResult,
)
from .task_space import ParameterSpec, TaskParameterSpace

__all__ = [
    "CurriculumDispatcher",
    "CurriculumRecord",
    "DesignerAction",
    "DesignerContext",
    "EvaluationResult",
    "IterationResult",
    "MATDDConfig",
    "MATDDRunResult",
    "MATDDTrainingLoop",
    "ParameterSpec",
    "PPOCurriculumDesigner",
    "RandomCurriculumDesigner",
    "TaskParameterSpace",
    "TrainingResult",
]
