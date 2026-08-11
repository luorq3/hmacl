"""Curriculum designers used by MATDD."""

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .task_space import Number, Task, TaskParameterSpace


@dataclass(frozen=True)
class DesignerContext:
    """Observable meta-state supplied to the curriculum designer."""

    current_task: Mapping[str, Number]
    target_task: Mapping[str, Number]
    progress: float
    target_return: float
    pool_fill_ratio: float

    def vector(
        self, space: TaskParameterSpace, return_scale: float = 1.0
    ) -> Tuple[float, ...]:
        if return_scale <= 0:
            raise ValueError("return_scale must be positive")
        return (
            space.normalize(self.current_task)
            + space.normalize(self.target_task)
            + (
                min(1.0, max(0.0, float(self.progress))),
                math.tanh(float(self.target_return) / return_scale),
                min(1.0, max(0.0, float(self.pool_fill_ratio))),
            )
        )


@dataclass(frozen=True)
class DesignerAction:
    task: Task
    normalized_task: Tuple[float, ...]
    metadata: Any = None


class RandomCurriculumDesigner:
    """Uniform designer used for tests and the domain-randomization baseline."""

    def __init__(
        self,
        space: TaskParameterSpace,
        target_task: Mapping[str, Number],
        seed: Optional[int] = None,
    ):
        self.space = space
        self.target_task = dict(target_task)
        self._rng = random.Random(seed)

    @property
    def pending_transitions(self) -> int:
        return 0

    def propose(self, context: DesignerContext) -> DesignerAction:
        normalized = tuple(self._rng.random() for _ in range(self.space.dimension))
        task = self.space.clip_to_target(
            self.space.denormalize(normalized), self.target_task
        )
        return DesignerAction(task=task, normalized_task=self.space.normalize(task))

    def observe(
        self, action: DesignerAction, reward: float, done: bool = False
    ) -> None:
        del action, reward, done

    def update(self, last_context: Optional[DesignerContext] = None):
        del last_context
        return {}

    def save(self, path: str) -> None:
        del path


class LinearCurriculumDesigner:
    """Linearly interpolate every free parameter from the initial task."""

    def __init__(
        self,
        space: TaskParameterSpace,
        initial_task: Mapping[str, Number],
        target_task: Mapping[str, Number],
    ):
        self.space = space
        self.initial_task = dict(initial_task)
        self.target_task = dict(target_task)
        self._initial_vector = space.normalize(initial_task)
        self._target_vector = space.normalize(target_task)

    @property
    def pending_transitions(self) -> int:
        return 0

    def propose(self, context: DesignerContext) -> DesignerAction:
        progress = min(1.0, max(0.0, float(context.progress)))
        normalized = tuple(
            initial + progress * (target - initial)
            for initial, target in zip(self._initial_vector, self._target_vector)
        )
        task = self.space.clip_to_target(
            self.space.denormalize(normalized), self.target_task
        )
        return DesignerAction(task=task, normalized_task=self.space.normalize(task))

    def observe(
        self, action: DesignerAction, reward: float, done: bool = False
    ) -> None:
        del action, reward, done

    def update(self, last_context: Optional[DesignerContext] = None):
        del last_context
        return {}

    def save(self, path: str) -> None:
        del path


class TargetTaskDesigner:
    """Always emit the target task for the non-curriculum baseline."""

    def __init__(
        self,
        space: TaskParameterSpace,
        target_task: Mapping[str, Number],
    ):
        self.space = space
        self.target_task = dict(target_task)

    @property
    def pending_transitions(self) -> int:
        return 0

    def propose(self, context: DesignerContext) -> DesignerAction:
        del context
        return DesignerAction(
            task=dict(self.target_task),
            normalized_task=self.space.normalize(self.target_task),
        )

    def observe(
        self, action: DesignerAction, reward: float, done: bool = False
    ) -> None:
        del action, reward, done

    def update(self, last_context: Optional[DesignerContext] = None):
        del last_context
        return {}

    def save(self, path: str) -> None:
        del path


class PPOCurriculumDesigner:
    """A bounded Beta-policy PPO teacher.

    PyTorch is imported lazily so the rest of the MATDD state machine can be
    tested without ML dependencies.  The teacher reward is supplied by the
    training loop and is normally parallel-team regret.
    """

    def __init__(
        self,
        space: TaskParameterSpace,
        target_task: Mapping[str, Number],
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 16,
        return_scale: float = 1.0,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError(
                "PPOCurriculumDesigner requires PyTorch; "
                "install project dependencies first"
            ) from exc
        if hidden_dim <= 0 or learning_rate <= 0 or minibatch_size <= 0:
            raise ValueError("invalid PPO designer hyperparameters")

        class ActorCritic(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                self.actor = nn.Linear(hidden_dim, output_dim * 2)
                self.critic = nn.Linear(hidden_dim, 1)

            def distribution_and_value(self, observations):
                features = self.body(observations)
                parameters = self.actor(features)
                alpha_raw, beta_raw = parameters.chunk(2, dim=-1)
                alpha = functional.softplus(alpha_raw) + 1.0
                beta = functional.softplus(beta_raw) + 1.0
                return torch.distributions.Beta(alpha, beta), self.critic(
                    features
                ).squeeze(-1)

        self._torch = torch
        self.space = space
        self.target_task = dict(target_task)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.return_scale = return_scale
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
        input_dim = space.dimension * 2 + 3
        self.model = ActorCritic(input_dim, space.dimension).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self._transitions = []

    @property
    def pending_transitions(self) -> int:
        return len(self._transitions)

    def _observation(self, context: DesignerContext):
        return self._torch.tensor(
            context.vector(self.space, self.return_scale),
            dtype=self._torch.float32,
            device=self.device,
        )

    def propose(self, context: DesignerContext) -> DesignerAction:
        observation = self._observation(context)
        with self._torch.no_grad():
            distribution, value = self.model.distribution_and_value(observation)
            action = distribution.sample().clamp(1e-6, 1.0 - 1e-6)
            log_probability = distribution.log_prob(action).sum(-1)
        normalized = tuple(float(item) for item in action.cpu().tolist())
        task = self.space.clip_to_target(
            self.space.denormalize(normalized), self.target_task
        )
        metadata = {
            "observation": observation.detach().cpu(),
            "action": action.detach().cpu(),
            "log_probability": float(log_probability.item()),
            "value": float(value.item()),
        }
        return DesignerAction(
            task=task,
            normalized_task=self.space.normalize(task),
            metadata=metadata,
        )

    def observe(
        self, action: DesignerAction, reward: float, done: bool = False
    ) -> None:
        if action.metadata is None:
            raise ValueError("PPO action metadata is missing")
        self._transitions.append(
            {
                **action.metadata,
                "reward": float(reward),
                "done": bool(done),
            }
        )

    def update(self, last_context: Optional[DesignerContext] = None):
        torch = self._torch
        if not self._transitions:
            return {}
        with torch.no_grad():
            last_value = 0.0
            if last_context is not None and not self._transitions[-1]["done"]:
                _, value = self.model.distribution_and_value(
                    self._observation(last_context)
                )
                last_value = float(value.item())

        advantages = []
        gae = 0.0
        next_value = last_value
        for transition in reversed(self._transitions):
            non_terminal = 0.0 if transition["done"] else 1.0
            delta = (
                transition["reward"]
                + self.gamma * next_value * non_terminal
                - transition["value"]
            )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.append(gae)
            next_value = transition["value"]
        advantages.reverse()
        returns = [
            advantage + transition["value"]
            for advantage, transition in zip(advantages, self._transitions)
        ]

        observations = torch.stack(
            [transition["observation"] for transition in self._transitions]
        ).to(self.device)
        actions = torch.stack(
            [transition["action"] for transition in self._transitions]
        ).to(self.device)
        old_log_probabilities = torch.tensor(
            [transition["log_probability"] for transition in self._transitions],
            dtype=torch.float32,
            device=self.device,
        )
        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(advantages) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std(unbiased=False) + 1e-8
            )

        losses = []
        sample_count = len(self._transitions)
        for _ in range(self.epochs):
            for indices in torch.randperm(sample_count, device=self.device).split(
                self.minibatch_size
            ):
                distribution, values = self.model.distribution_and_value(
                    observations[indices]
                )
                log_probabilities = distribution.log_prob(actions[indices]).sum(-1)
                ratio = (log_probabilities - old_log_probabilities[indices]).exp()
                unclipped = ratio * advantages_tensor[indices]
                clipped = (
                    ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
                    * advantages_tensor[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = (returns_tensor[indices] - values).pow(2).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                losses.append(float(loss.item()))
        self._transitions.clear()
        return {"teacher_loss": sum(losses) / len(losses)}

    def save(self, path: str) -> None:
        self._torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = self._torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
