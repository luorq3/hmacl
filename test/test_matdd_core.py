import unittest

try:
    import numpy as np
    import torch  # noqa: F401
except ImportError:
    ML_DEPENDENCIES_AVAILABLE = False
else:
    ML_DEPENDENCIES_AVAILABLE = True

if ML_DEPENDENCIES_AVAILABLE:
    from hmacl.matdd.adapters.padded_env import PaddedMultiAgentEnv
else:
    PaddedMultiAgentEnv = None
from hmacl.matdd.designer import (
    DesignerContext,
    PPOCurriculumDesigner,
    RandomCurriculumDesigner,
)
from hmacl.matdd.dispatcher import CurriculumDispatcher
from hmacl.matdd.loop import (
    EvaluationResult,
    MATDDConfig,
    MATDDTrainingLoop,
    TrainingResult,
)
from hmacl.matdd.task_space import ParameterSpec, TaskParameterSpace


class FakeStudent:
    def __init__(self, advantage=0.0):
        self.advantage = advantage
        self.train_calls = []

    def train(self, task, step_budget):
        self.train_calls.append((dict(task), step_budget))
        score = float(task["size"]) + self.advantage
        return TrainingResult(
            mean_return=score,
            mean_abs_td_error=1.0 / score,
            environment_steps=step_budget,
        )

    def evaluate(self, task, episodes):
        return EvaluationResult(
            mean_return=float(task["size"]),
            environment_steps=episodes,
        )


class TaskParameterSpaceTest(unittest.TestCase):
    def test_round_trip_and_distance(self):
        space = TaskParameterSpace(
            [
                ParameterSpec("size", 10, 50, integer=True),
                ParameterSpec("agents", 5, 25, integer=True),
            ]
        )
        self.assertEqual(
            space.denormalize(space.normalize({"size": 30, "agents": 15})),
            {"size": 30, "agents": 15},
        )
        self.assertAlmostEqual(
            space.distance({"size": 10, "agents": 5}, {"size": 50, "agents": 25}),
            2**0.5,
        )


class DispatcherTest(unittest.TestCase):
    def setUp(self):
        self.space = TaskParameterSpace([ParameterSpec("size", 1, 10, integer=True)])

    def test_mixture_prefers_value_and_target_proximity(self):
        dispatcher = CurriculumDispatcher(
            self.space, {"size": 10}, capacity=3, rho=0.5, temperature=1.0, seed=3
        )
        dispatcher.observe({"size": 2}, training_value=10.0)
        dispatcher.observe({"size": 6}, training_value=2.0)
        dispatcher.observe({"size": 9}, training_value=1.0)
        probabilities = dispatcher.probabilities()
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(probabilities[0], probabilities[2])
        self.assertLess(probabilities[1], probabilities[0])

    def test_replay_updates_without_removing_and_fifo_evicts(self):
        dispatcher = CurriculumDispatcher(self.space, {"size": 10}, capacity=2, seed=1)
        dispatcher.observe({"size": 2}, 1.0)
        dispatcher.observe({"size": 4}, 2.0)
        dispatcher.observe({"size": 4}, 3.0)
        self.assertEqual(len(dispatcher), 2)
        self.assertEqual(dispatcher.records[1].visits, 2)
        dispatcher.observe({"size": 6}, 4.0)
        self.assertEqual([record.task["size"] for record in dispatcher.records], [4, 6])


class MATDDTrainingLoopTest(unittest.TestCase):
    def test_loop_accounts_for_both_students_and_evaluation(self):
        space = TaskParameterSpace([ParameterSpec("size", 1, 10, integer=True)])
        protagonist = FakeStudent(advantage=0.0)
        antagonist = FakeStudent(advantage=2.0)
        designer = RandomCurriculumDesigner(space, {"size": 10}, seed=4)
        dispatcher = CurriculumDispatcher(space, {"size": 10}, capacity=2, seed=5)
        config = MATDDConfig(
            curriculum_iterations=3,
            steps_per_curriculum=20,
            final_target_steps=30,
            evaluation_episodes=4,
            seed=6,
        )
        result = MATDDTrainingLoop(
            protagonist,
            antagonist,
            designer,
            dispatcher,
            initial_task={"size": 1},
            target_task={"size": 10},
            config=config,
        ).run()

        self.assertEqual(len(result.iterations), 3)
        self.assertTrue(
            all(item.parallel_team_regret == 2.0 for item in result.iterations)
        )
        self.assertEqual(
            [item.source for item in result.iterations],
            ["designer", "designer", "dispatcher"],
        )
        expected_steps = 4 + 3 * (20 + 20 + 4) + 30
        self.assertEqual(result.total_environment_steps, expected_steps)
        self.assertEqual(len(dispatcher), 2)


@unittest.skipUnless(ML_DEPENDENCIES_AVAILABLE, "ML dependencies are not installed")
class PPOCurriculumDesignerTest(unittest.TestCase):
    def test_designer_updates_and_clears_rollout(self):
        space = TaskParameterSpace([ParameterSpec("size", 1, 10, integer=True)])
        designer = PPOCurriculumDesigner(
            space,
            {"size": 10},
            hidden_dim=16,
            epochs=1,
            minibatch_size=2,
            seed=7,
        )
        context = DesignerContext(
            current_task={"size": 1},
            target_task={"size": 10},
            progress=0.0,
            target_return=0.0,
            pool_fill_ratio=0.0,
        )
        for reward in (-1.0, 0.0, 1.0, 2.0):
            action = designer.propose(context)
            designer.observe(action, reward)
        metrics = designer.update(context)
        self.assertIn("teacher_loss", metrics)
        self.assertEqual(designer.pending_transitions, 0)


@unittest.skipUnless(ML_DEPENDENCIES_AVAILABLE, "NumPy is not installed")
class PaddedMultiAgentEnvTest(unittest.TestCase):
    class FakeEnv:
        def __init__(self):
            self.n_agents = 2
            self.n_actions = 3
            self.episode_limit = 5
            self.last_actions = None

        def get_env_info(self):
            return {
                "n_agents": self.n_agents,
                "n_actions": self.n_actions,
                "obs_shape": 2,
                "state_shape": 4,
                "episode_limit": self.episode_limit,
            }

        def get_obs(self):
            return np.array([[1, 2], [3, 4]], dtype=np.float32)

        def get_obs_agent(self, agent_id):
            return self.get_obs()[agent_id]

        def get_state(self):
            return np.array([1, 2, 3, 4], dtype=np.float32)

        def get_avail_actions(self):
            return np.ones((self.n_agents, self.n_actions), dtype=np.int64)

        def get_classes(self):
            return np.array([0, 1])

        def step(self, actions):
            self.last_actions = list(actions)
            return 0.0, True, {}

        def reset(self):
            return None

        def get_stats(self):
            return {}

        def render(self):
            return None

        def close(self):
            return None

        def seed(self, seed=None):
            return seed

        def save_replay(self):
            return None

    def test_padding_and_action_truncation(self):
        raw_env = self.FakeEnv()
        env = PaddedMultiAgentEnv(
            raw_env,
            {
                "n_agents": 4,
                "n_actions": 5,
                "obs_shape": 3,
                "state_shape": 6,
                "episode_limit": 10,
            },
        )
        self.assertEqual(env.get_obs().shape, (4, 3))
        self.assertEqual(env.get_state().shape, (6,))
        available = env.get_avail_actions()
        self.assertEqual(available.shape, (4, 5))
        self.assertTrue(np.all(available[2:, 0] == 1))
        self.assertTrue(np.all(available[2:, 1:] == 0))
        self.assertEqual(env.get_agent_mask().tolist(), [[1.0], [1.0], [0.0], [0.0]])
        env.step([0, 1, 2, 3])
        self.assertEqual(raw_env.last_actions, [0, 1])


if __name__ == "__main__":
    unittest.main()
