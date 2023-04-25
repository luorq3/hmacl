import numpy as np
from src.beachlanding.scenario_builder import build_scenario


class CombatEnvCore:
    def __init__(self, config):
        self._config = config
        self._init()

    def _init(self):
        self._scenario = build_scenario(self._config)

        self.map_size = self._scenario.map_size
        self.surface_map = self._scenario.surface_map
        self.air_map = self._scenario.air_map

        # agents need to control by algo, units includes both [agents] and [bot]
        self.agents_unique_idx = self._scenario.offensive_agents_id + self._scenario.defensive_agents_id
        self.bots_unique_idx = self._scenario.offensive_bots_id + self._scenario.defensive_bots_id
        self.state_dims = self._scenario.state_dims
        self.state_spaces = self._scenario.state_spaces
        # [2,4, 3,5,..., 4,7,]
        # agents dict
        self.units_dict = self._scenario.units_dict
        for unit in self.units_dict.values():
            unit.update_visible_units(self.units_dict)
            unit.update_avail_act(self.units_dict, self.surface_map, self.air_map)
            # print([unit.unique_id, unit.x, unit.y, unit.sight_range], unit.visible_units)

        # reward
        self.reward_type = self._scenario.reward_type

    def reset(self):
        self._init()

    def get_agent(self, unique_id):
        return self.units_dict[unique_id]

    def get_obs_dim(self):
        obs_dims = {unique_id: self.units_dict.get(unique_id).obs_dim for unique_id in self.agents_unique_idx}
        return obs_dims

    def get_obs(self):
        obs = {unique_id: self.units_dict.get(unique_id).obs(self.units_dict) for unique_id in self.agents_unique_idx}
        return obs

    def get_obs_unique_id(self, unique_id):
        return self.units_dict.get(unique_id).obs(self.units_dict)

    def get_state(self):
        # position of all army,position of all enemy, HP of all army, HP of all enemy
        pos = np.array([[-1, -1] if unit.dead else [unit.x, unit.y] for unit in self.units_dict.values()]).flatten()
        hp = np.array([-1 if unit.dead else unit.hp for unit in self.units_dict.values()]).flatten()
        state = np.concatenate((pos, hp))
        return state

    def get_avail_actions(self):
        avail_acts = {unique_id: self.units_dict.get(unique_id).get_avail_act() for unique_id in self.agents_unique_idx}
        return avail_acts

    def get_agent_avail_actions(self, unique_id):
        agent = self.units_dict.get(unique_id)
        return agent.get_avail_act()

    def step(self, actions):
        assert len(self.agents_unique_idx) == len(actions), "Actions can't match agents."
        for unique_id, action in actions.items():
            agent = self.units_dict.get(unique_id)
            agent.set_action(action, self.units_dict, self.surface_map, self.air_map)

        for unique_id in self.bots_unique_idx:
            bot = self.units_dict.get(unique_id)
            bot.set_action_for_bot(self.units_dict, self.surface_map, self.air_map)

        info = {}
        rewards = {}
        for unit in self.units_dict.values():
            rewards[unit.unique_id] = unit.take_action(self.units_dict, self.surface_map, self.air_map)

        for unit in self.units_dict.values():
            unit.update_visible_units(self.units_dict)
            unit.update_avail_act(self.units_dict, self.surface_map, self.air_map)
            # print([unit.unique_id, unit.x, unit.y, unit.sight_range, unit.hp], unit.visible_units)

        of_dones = []
        de_dones = []
        for unit in self.units_dict.values():
            if unit.camp == "offensive":
                of_dones.append(unit.dead)
            else:
                de_dones.append(unit.dead)
        of_done = np.all(of_dones)
        de_done = np.all(de_dones)

        done = of_done or de_done
        if self.reward_type == "dense":
            reward = rewards
        elif self.reward_type == "sparse":
            reward = {}
            if done:
                reward["offensive"] = float(de_done)
                reward["defensive"] = float(of_done)
            else:
                reward = {"offensive": 0., "defensive": 0.}
        else:
            raise ValueError(f"Doesn't exists reward_type: {self.reward_type}")

        if done:
            info = {
                "of_win": 0,
                "de_win": 0
            }
            if of_done:
                info["de_win"] = 1
            if de_done:
                info["of_win"] = 1

        return reward, done, info
