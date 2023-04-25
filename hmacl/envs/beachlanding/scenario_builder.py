from src import logging
import random
import sys

import numpy as np
from gym import spaces

from src.beachlanding.util import load_scenario_conf
from src.beachlanding.agent import *


def build_scenario(config):
    try:
        scenario_conf = load_scenario_conf(config)
    except FileNotFoundError as e:
        logging.error('Loading scenario "%s" failed' % config['scenario'])
        logging.error(e)
        sys.exit(1)

    return Scenario(scenario_conf)


class Scenario(object):
    def __init__(self, config):
        self._scenario_cfg = config

        self.map_size = config.map_size
        self.pos_init_type = config.pos_init_type  # initialize position, "random", "boundary" or "specified"
        self.reward_type = config.reward_type  # "dense" or "sparse"
        self.obs_with_agent_id = config.obs_with_agent_id  # True or False

        self.surface_map = np.zeros(self.map_size) - 1
        self.air_map = np.zeros(self.map_size) - 1

        self.units_dict = {}
        self.offensive_agents_id = []
        self.offensive_bots_id = []
        self.defensive_agents_id = []
        self.defensive_bots_id = []

        self._init_agents()
        self._setting_agents()

    def _setting_agents(self):
        self.state_dims = {}
        self.state_spaces = {}
        for camp in CAMP_KEYS:
            self._setting_camp_agents(camp)

    def _setting_camp_agents(self, camp):
        if camp == "offensive":
            units = [self.units_dict[unique_id] for unique_id in self.offensive_agents_id + self.offensive_bots_id]
            num_allies = self.num_offensive_agents + self.num_offensive_bots
            num_rivals = self.num_defensive_agents + self.num_defensive_bots
            num_allies_agents = self.num_offensive_agents
        else:
            units = [self.units_dict[unique_id] for unique_id in self.defensive_agents_id + self.defensive_bots_id]
            num_allies = self.num_defensive_agents + self.num_defensive_bots
            num_rivals = self.num_offensive_agents + self.num_offensive_bots
            num_allies_agents = self.num_defensive_agents

        for unit in units:
            unit.set_property(num_rivals, num_allies, num_allies_agents, self.obs_with_agent_id)
        state_dim = (num_allies + num_rivals) * 3
        self.state_dims[camp] = state_dim
        self.state_spaces[camp] = spaces.Box(low=np.inf, high=np.inf, shape=(state_dim, ), dtype=np.float32)

    def _init_agents(self):
        of_agents = self._init_camp_agents("offensive", "agent")
        of_bots = self._init_camp_agents("offensive", "bot")

        self.num_offensive_agents = len(of_agents)
        self.num_offensive_bots = len(of_bots)
        self.num_offensive_units = self.num_offensive_agents + self.num_offensive_bots

        for unique_id, of_agent in enumerate(of_agents):
            of_agent.unique_id = unique_id
            self.units_dict[of_agent.unique_id] = of_agent
            self.offensive_agents_id.append(of_agent.unique_id)
        for unique_id, of_bot in enumerate(of_bots):
            of_bot.unique_id = unique_id + self.num_offensive_agents
            self.units_dict[of_bot.unique_id] = of_bot
            self.offensive_bots_id.append(of_bot.unique_id)

        df_agents = self._init_camp_agents("defensive", "agent")
        df_bots = self._init_camp_agents("defensive", "bot")
        self.num_defensive_agents = len(df_agents)
        self.num_defensive_bots = len(df_bots)
        self.num_defensive_units = self.num_defensive_agents + self.num_defensive_bots
        for unique_id, df_agent in enumerate(df_agents):
            df_agent.unique_id = unique_id + self.num_offensive_units
            self.units_dict[df_agent.unique_id] = df_agent
            self.defensive_agents_id.append(df_agent.unique_id)
        for unique_id, df_bot in enumerate(df_bots):
            df_bot.unique_id = unique_id + self.num_offensive_units + self.num_defensive_agents
            self.units_dict[df_bot.unique_id] = df_bot
            self.defensive_bots_id.append(df_bot.unique_id)

        self._init_position("offensive", of_agents + of_bots)
        self._init_position("defensive", df_agents + df_bots)

    def _init_camp_agents(self, camp, control_type):
        agents = []
        conf = getattr(self._scenario_cfg, camp + "_" + control_type, None)
        if conf is None:
            return agents

        for agent_type in AGENT_KEYS:
            agents_conf = getattr(conf, agent_type, None)
            if not isinstance(agents_conf, list) or len(agents_conf) == 0:
                continue
            agent_class = AGENT_CLASS_DICT.get(agent_type)
            for i, agent_conf in enumerate(agents_conf):
                new_agent_conf = self._check_camp_units_conf(camp, agent_type, control_type, i, agent_conf)
                agent = agent_class(**new_agent_conf)
                agents.append(agent)

        return agents

    def _init_position(self, camp, units):
        if self.pos_init_type == "specified":
            for unit in units:
                if unit.map_type == "air":
                    e_map = self.air_map
                else:
                    e_map = self.surface_map
                e_map[unit.x, unit.y] = unit.unique_id
            return

        if camp == "offensive":
            y_start = 0
        else:
            y_start = self.map_size[1] - 2
        if self.pos_init_type == "random":
            # x-random y-boundary
            pos_gen_fn = self._generate_random_pos
        elif self.pos_init_type == "boundary":
            # x-average y-boundary
            pos_gen_fn = self._generate_average_pos
        else:
            raise ValueError(f"No such pos_init_type: {self.pos_init_type}")
        num_units = len(units)
        for unit in units:
            if unit.map_type == "air":
                e_map = self.air_map
            else:
                e_map = self.surface_map
            x, y = pos_gen_fn(e_map, y_start, num_units)
            e_map[x, y] = unit.unique_id
            unit.x = x
            unit.y = y

    def _generate_average_pos(self, e_map, y_start, num_units):
        raise NotImplementedError

    def _generate_random_pos(self, e_map, y_start, num_units):
        x = random.randint(0, self.map_size[1] - 1)
        y = random.randint(y_start, y_start + 1)
        if e_map[x, y] >= 0:
            x, y = self._generate_random_pos(e_map, y_start, num_units)
        return x, y

    def _check_camp_units_conf(self, camp, unit_type, control_type, i, unit_conf):
        new_conf = {
            "control_type": control_type,
            "agent_type": unit_type,
            "camp": camp,
            "hp": unit_conf['hp'],
            "sight_range": unit_conf['sight_range'],
            "map_size": self.map_size}

        if unit_conf.get("dmg") is not None:
            new_conf["dmg"] = unit_conf["dmg"]

        # boundary: 从地图最边缘开始，均匀交错排列, 最多排列两列
        if self.pos_init_type == "boundary":  # 一方的单位数量不可以超过
            pass
        # random: 在已方阵营随机初始化，一旦随机选择的位置已经被占用，则重新随机
        elif self.pos_init_type == "random":
            pass
        # specified: 指定一个位置
        elif self.pos_init_type == "specified":
            pos = unit_conf.get('pos')
            if unit_conf.get('pos') is None or not isinstance(pos, list):
                raise TypeError(f"Specify a position for {camp}-{unit_type}-{str(i)}")
            new_conf['x'] = pos[0]
            new_conf['y'] = pos[1]
        else:
            raise TypeError(f"Unrecognized pos_init_type: [{self.pos_init_type}]")
        return new_conf
