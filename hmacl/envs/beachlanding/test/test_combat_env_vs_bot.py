import unittest
from argparse import ArgumentParser

import numpy as np

from src.beachlanding.combat_env_vs_bot import CombatEnv


parser = ArgumentParser()
parser.add_argument("--scenario", type=str, default="agent_vs_bot_example")
parser.add_argument("--camp", type=str, default="offensive")
parser.add_argument("--reward_type", type=str, default="dense")
config = parser.parse_args()

env = CombatEnv(**dict(config))
env.reset()
agents = env.agent_idx_dict


class TestEnv(unittest.TestCase):

    def test_get_obs(self):
        obs = env.get_obs()
        print(obs)

    def test_get_obs_agent(self):
        obs = env.get_obs()
        self.assertTrue(np.all(np.equal(obs[0], env.get_obs_agent(0))))
        self.assertTrue(np.all(np.equal(obs[1], env.get_obs_agent(1))))
        self.assertTrue(np.all(np.equal(obs[2], env.get_obs_agent(2))))

    def test_get_obs_size(self):
        self.assertEqual(env.get_obs_size(), [18, 18, 18])

    def test_get_state(self):
        state = np.array([3, 1, 0, 0, 5, 0, 5, 5, 2, 4, 0, 5, 10, 8, 5, 8, 10, 5])
        self.assertTrue(np.all(np.equal(env.get_state(), state)))

    def test_get_state_size(self):
        state_dim = 18
        self.assertEqual(state_dim, env.get_state_size())

    def test_get_avail_actions(self):
        # array([1., 1., 1., 0., 1., 0.]),
        # array([1., 0., 1., 0., 1., 0., 0., 0.])
        # array([1., 1., 0., 0., 1., 1., 0., 0.])
        print(env.get_avail_actions())

    def test_get_avail_agent_actions(self):
        print(env.get_avail_agent_actions(0))
        print(env.get_avail_agent_actions(1))
        print(env.get_avail_agent_actions(2))

    def test_fighter_move(self):
        print(env._env.air_map)

        actions = [0, 4, 0]
        reward, done, info = env.step(actions)
        # print(reward, done, info)
        # print(env.get_avail_agent_actions(0))
        print(env.get_avail_agent_actions(1))
        # self.assertTrue(np.all(np.equal(env.get_avail_agent_actions(1), [1., 0., 1., 1., 1., 0., 0., 0.])))

        actions = [0, 4, 0]
        reward, done, info = env.step(actions)
        # print(reward, done, info)
        # print(env.get_avail_agent_actions(0))
        print(env.get_avail_agent_actions(1))
        # self.assertTrue(np.all(np.equal(env.get_avail_agent_actions(1), [1., 0., 1., 1., 1., 0., 1., 0.])))

        actions = [0, 4, 0]
        reward, done, info = env.step(actions)
        # print(reward, done, info)
        # print(env.get_avail_agent_actions(0))
        print(env.get_avail_agent_actions(1))
        # self.assertTrue(np.all(np.equal(env.get_avail_agent_actions(1), [1., 0., 1., 1., 1., 0., 1., 1.])))
        # print(env._env.get_agent(4).hp)

        actions = [0, 4, 0]
        reward, done, info = env.step(actions)
        # print(reward, done, info)
        # print(env.get_avail_agent_actions(0))
        print(env.get_avail_agent_actions(1))
        # self.assertTrue(np.all(np.equal(env.get_avail_agent_actions(1), [1., 0., 1., 1., 0., 0., 1., 1.])))
        # print(env._env.get_agent(4).hp)

        print(env._env.air_map)

    def test_fighter_fire_and_share(self):
        # actions = [0, 4, 0]
        # reward, done, info = env.step(actions)
        # print(reward, done, info)
        # # actions = [0, 4, 0]
        # # reward, done, info = env.step(actions)
        # # print(reward, done, info)
        print(env.get_avail_agent_actions(1))
        print(env.get_avail_agent_actions(2))
        actions = [0, 0, 6]
        reward, done, info = env.step(actions)
        print(env.get_avail_agent_actions(1))
        print(env.get_unit_by_unique_id(1))
        print(env.get_unit_by_unique_id(4))
        # self.assertTrue(np.all(np.equal(env.get_avail_agent_actions(1), [1., 0., 1., 1., 1., 0., 1., 0.])))
        # for _ in range(3):
        #     actions = [0, 6, 0]
        #     reward, done, info = env.step(actions)
        #     print(reward, done, info)
        #     print(env.get_avail_agent_actions(1))
        #     print(env.get_unit_by_unique_id(1))
        #     print(env.get_unit_by_unique_id(4))
        #     self.assertTrue(np.all(np.equal(env.get_avail_agent_actions(1), [1., 0., 1., 1., 1., 0., 1., 0.])))

        actions = [0, 6, 0]
        reward, done, info = env.step(actions)
        print(reward, done, info)
        print(env.get_avail_agent_actions(1))
        print(env.get_unit_by_unique_id(1))
        print(env.get_unit_by_unique_id(4))
        # self.assertTrue(np.all(np.equal(env.get_avail_agent_actions(1), [1., 0., 1., 1., 1., 0., 0., 0.])))

    def test_step_inf(self):
        for episode in range(10):
            done = False
            i = 0
            while not done:
                aa1 = np.random.choice(np.nonzero(env.get_avail_agent_actions(0))[0])
                aa2 = np.random.choice(np.nonzero(env.get_avail_agent_actions(1))[0])
                aa3 = np.random.choice(np.nonzero(env.get_avail_agent_actions(2))[0])
                actions = [aa1, aa2, aa3]
                # actions = [aa1, aa2]
                reward, done, info = env.step(actions)
                print(f"episode={episode}, step={i}, reward={reward}, done={done}, info={info}")
                i += 1
            env.reset()
