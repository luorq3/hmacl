import unittest

try:
    import numpy as np
    import torch  # noqa: F401
except ImportError:
    ML_DEPENDENCIES_AVAILABLE = False
else:
    ML_DEPENDENCIES_AVAILABLE = True

try:
    import vmas  # noqa: F401
except ImportError:
    VMAS_AVAILABLE = False
else:
    VMAS_AVAILABLE = True

try:
    from pettingzoo.sisl import pursuit_v4  # noqa: F401
except ImportError:
    PETTINGZOO_AVAILABLE = False
else:
    PETTINGZOO_AVAILABLE = True

if ML_DEPENDENCIES_AVAILABLE:
    from hmacl.algo.envs.pursuit import PettingZooPursuitEnv
    from hmacl.algo.envs.vmas import VMASFootballEnv
    from hmacl.algo.modules.agents.matdd_conv_agent import MATDDConvAgent
    from hmacl.algo.modules.agents.matdd_rnn_agent import MATDDRNNAgent
    from hmacl.algo.modules.critics.dyna_central_v import DynaCentralVCritic
    from hmacl.algo.modules.mixers.dyna_qmix import DynaQMixer
    from hmacl.matdd.adapters.football import FootballTaskAdapter
    from hmacl.matdd.adapters.padded_env import PaddedMultiAgentEnv
    from hmacl.matdd.adapters.pursuit import PursuitTaskAdapter
else:
    FootballTaskAdapter = None
    MATDDConvAgent = None
    MATDDRNNAgent = None
    DynaCentralVCritic = None
    DynaQMixer = None
    PettingZooPursuitEnv = None
    PaddedMultiAgentEnv = None
    PursuitTaskAdapter = None
    VMASFootballEnv = None
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
            metrics={"win_rate": float(task["size"]) / 10.0},
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


@unittest.skipUnless(ML_DEPENDENCIES_AVAILABLE, "NumPy is not installed")
class FootballTaskAdapterTest(unittest.TestCase):
    def test_scales_both_teams_and_removes_adapter_only_keys(self):
        adapter = FootballTaskAdapter(
            {
                "scenario": "football",
                "n_blue_agents": 3,
                "n_red_agents": 3,
                "target_n_agents": 8,
                "scale_red_team": True,
            }
        )
        config = adapter.env_config({"n_agents": 5})
        self.assertEqual(config["n_blue_agents"], 5)
        self.assertEqual(config["n_red_agents"], 5)
        self.assertNotIn("target_n_agents", config)
        self.assertNotIn("scale_red_team", config)


@unittest.skipUnless(ML_DEPENDENCIES_AVAILABLE, "NumPy is not installed")
class PursuitTaskAdapterTest(unittest.TestCase):
    def test_maps_two_normalized_parameters_to_square_pursuit(self):
        adapter = PursuitTaskAdapter(
            {
                "x_size": 16,
                "y_size": 16,
                "n_pursuers": 8,
                "n_evaders": 30,
                "target_map_size": 60,
                "target_n_pursuers": 20,
                "scale_n_evaders": False,
            }
        )
        config = adapter.env_config({"map_size": 40, "n_pursuers": 14})
        self.assertEqual(config["x_size"], 40)
        self.assertEqual(config["y_size"], 40)
        self.assertEqual(config["n_pursuers"], 14)
        self.assertEqual(config["n_evaders"], 30)
        self.assertEqual(adapter.space.dimension, 2)


@unittest.skipUnless(ML_DEPENDENCIES_AVAILABLE, "PyTorch is not installed")
class MATDDRNNAgentTest(unittest.TestCase):
    def test_matches_paper_layer_dimensions(self):
        from types import SimpleNamespace

        agent = MATDDRNNAgent(
            20,
            SimpleNamespace(
                n_actions=9,
                matdd_encoder_dim=256,
                matdd_recurrent_dim=256,
                matdd_projection_dim=128,
            ),
        )
        output, hidden = agent(torch.zeros(6, 20), agent.init_hidden().repeat(6, 1))
        self.assertEqual(output.shape, (6, 9))
        self.assertEqual(hidden.shape, (6, 256))
        self.assertEqual(agent.fc2.out_features, 128)


@unittest.skipUnless(ML_DEPENDENCIES_AVAILABLE, "PyTorch is not installed")
class MATDDConvAgentTest(unittest.TestCase):
    def test_extracts_144_spatial_features_before_recurrent_stack(self):
        from types import SimpleNamespace

        agent = MATDDConvAgent(
            147 + 20,
            SimpleNamespace(
                n_actions=5,
                obs_shape=(7, 7, 3),
                matdd_conv_channels=16,
                matdd_encoder_dim=256,
                matdd_recurrent_dim=256,
                matdd_projection_dim=128,
            ),
        )
        output, hidden = agent(torch.zeros(6, 167), agent.init_hidden().repeat(6, 1))
        self.assertEqual(output.shape, (6, 5))
        self.assertEqual(hidden.shape, (6, 256))
        self.assertEqual(agent.fc1.in_features, 144 + 20)


@unittest.skipUnless(ML_DEPENDENCIES_AVAILABLE, "PyTorch is not installed")
class DynaNetworkTest(unittest.TestCase):
    class FakeBatch:
        def __init__(self, obs, agent_mask):
            self.data = {"obs": obs, "agent_mask": agent_mask}
            self.scheme = {"obs": {}, "agent_mask": {}}
            self.batch_size = obs.shape[0]
            self.max_seq_length = obs.shape[1]
            self.device = obs.device

        def __getitem__(self, key):
            return self.data[key]

    def test_qmix_ignores_masked_agent_content(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            n_agents=4,
            obs_shape=(2,),
            mixing_embed_dim=8,
            hypernet_embed=16,
            dyna_hidden_dim=8,
            dyna_embedding_dim=6,
        )
        mixer = DynaQMixer(args)
        observations = torch.randn(1, 3, 4, 2)
        agent_qs = torch.randn(1, 3, 4)
        mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]] * 3])
        changed_obs = observations.clone()
        changed_obs[:, :, 2:] = 1000.0
        changed_qs = agent_qs.clone()
        changed_qs[:, :, 2:] = -1000.0
        first = mixer(agent_qs, observations=observations, agent_mask=mask)
        second = mixer(changed_qs, observations=changed_obs, agent_mask=mask)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_central_critic_ignores_masked_agent_observations(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            n_agents=4,
            hidden_dim=16,
            dyna_hidden_dim=8,
            dyna_embedding_dim=6,
        )
        critic = DynaCentralVCritic({"obs": {"vshape": (2,)}}, args)
        observations = torch.randn(1, 3, 4, 2)
        mask = torch.tensor([[[[1.0], [1.0], [0.0], [0.0]]] * 3])
        changed = observations.clone()
        changed[:, :, 2:] = 1000.0
        first = critic(self.FakeBatch(observations, mask))[:, :, :2]
        second = critic(self.FakeBatch(changed, mask))[:, :, :2]
        self.assertTrue(torch.allclose(first, second, atol=1e-6))


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
        self.assertEqual(result.iterations[-1].target_metrics["win_rate"], 1.0)


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


@unittest.skipUnless(
    ML_DEPENDENCIES_AVAILABLE and VMAS_AVAILABLE,
    "VMAS dependencies are not installed",
)
class VMASFootballEnvTest(unittest.TestCase):
    def test_discrete_environment_rebuilds_for_curriculum(self):
        base_config = {
            "scenario": "football",
            "device": "cpu",
            "continuous_actions": False,
            "max_steps": 2,
            "seed": 9,
            "ai_red_agents": True,
            "ai_blue_agents": False,
            "n_blue_agents": 2,
            "n_red_agents": 2,
            "n_traj_points": 0,
            "dense_reward": True,
        }
        env = VMASFootballEnv(**base_config)
        try:
            env.reset()
            self.assertEqual(env.get_obs().shape[0], 2)
            _, terminated, _ = env.step([0, 0])
            self.assertFalse(terminated)
            _, terminated, info = env.step([0, 0])
            self.assertTrue(terminated)
            self.assertTrue(info["episode_limit"])

            updated_config = dict(base_config, n_blue_agents=3, n_red_agents=3)
            env.update(updated_config)
            env.reset()
            self.assertEqual(env.n_agents, 3)
            self.assertEqual(env.get_obs().shape[0], 3)
        finally:
            env.close()


@unittest.skipUnless(
    ML_DEPENDENCIES_AVAILABLE and PETTINGZOO_AVAILABLE,
    "PettingZoo Pursuit dependencies are not installed",
)
class PettingZooPursuitEnvTest(unittest.TestCase):
    def test_spatial_state_and_population_rebuild(self):
        base_config = {
            "map_name": "sisl",
            "scenario": "pursuit",
            "seed": 11,
            "max_cycles": 2,
            "x_size": 16,
            "y_size": 16,
            "shared_reward": True,
            "n_evaders": 6,
            "n_pursuers": 4,
            "obs_range": 7,
            "n_catch": 2,
            "freeze_evaders": False,
            "tag_reward": 0.01,
            "catch_reward": 5.0,
            "urgency_reward": -0.1,
            "surround": True,
            "constraint_window": 1.0,
        }
        env = PettingZooPursuitEnv(**base_config)
        try:
            env.reset()
            self.assertEqual(env.get_obs().shape, (4, 7, 7, 3))
            self.assertEqual(env.get_state().shape, (16, 16, 3))
            env.step([4] * 4)
            _, terminated, info = env.step([4] * 4)
            self.assertTrue(terminated)
            self.assertTrue(info["episode_limit"])

            env.update(dict(base_config, x_size=20, y_size=20, n_pursuers=5))
            env.reset()
            self.assertEqual(env.get_obs().shape, (5, 7, 7, 3))
            self.assertEqual(env.get_state().shape, (20, 20, 3))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
