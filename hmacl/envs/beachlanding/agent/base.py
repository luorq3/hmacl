"""
0: warship
1: fighter
2: mobile fort
3: scout drone
"""
from src.beachlanding.agent.util import merge_visible_units
from gym import spaces
import numpy as np


CONTROL_TYPES = {"agent": 0, "bot": 1}
CONTROL_KEYS = CONTROL_TYPES.keys()

AGENTS_TYPES = {"warship": 0, "fighter": 1, "mobile_fort": 2, "scout_drone": 3}
AGENT_KEYS = AGENTS_TYPES.keys()

CAMPS_TYPES = {"offensive": 0, "defensive": 1}
CAMP_KEYS = CAMPS_TYPES.keys()

ACTION_SET = ['noop', 'up', 'down', 'left', 'right', 'fire', 'share']


class AgentBase:
    def __init__(self, **kwargs):
        # control_type, agent_type, camp, x, y, hp, map_size, sight_range
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.unique_id = None

        # Just for agent of algorithm, decoupling with environment.
        self.agent_id = None

        # Other setting for agents
        self.action_dim = None
        self.advanced_action_dim = None
        self.action_space = None
        self.obs_dim = None
        self.obs_space = None
        self.num_units = None

        # To a "fire" action, if unit's camp is offensive, it should plus num_offensive_units to choose a defensive unit by unique_id.
        # To a "share" action, the same if unit's camp is defensive
        self.advanced_action_offset = None

        self._set_moving_boundary()

        # Square range based on position of itself. A unit can fire another enemy i.f.f can seen the enemy.
        self.visible_units = []
        self.received_info_from = []

        # avail_acts
        self.avail_acts = []

        self._action = {
            "type": None,  # 动作类型：base(include noop), advanced
            "real_val": None  # 动作真值: 1."base": action,
            # 2."advanced": [x, y, z] (x,y)=攻击的坐标,记录地图种类0=surface,1=air
            # 对于scout_drone需要选择一名友军分享信息，real_val=ally.unique_id
        }
        self._obs = None

    def __str__(self):
        properties = {
            "unique_id": self.unique_id,
            "hp": self.hp,
            "dmg": self.dmg,
            "pos": [self.x, self.y],
            "sight_range": self.sight_range
        }
        return str(properties)

    def _set_moving_boundary(self):
        # setting moving boundary
        self.min_x = 0
        self.min_y = 0
        self.max_x = self.map_size[0] - 1
        self.max_y = self.map_size[1] - 1

    # 对于移动类动作，在set_action时执行；对于攻击类动作，先记录被攻击坐标，在take_action时执行
    def set_action(self, action, unit_dict, surface_map, air_map):
        avail_acts = self.get_avail_act()
        assert float(avail_acts[action]), "Action is not avail."

        if action < self.base_action_dim:
            map_ = surface_map if self.map_type == "surface" else air_map
            self._action = {"type": "base", "real_val": action}
            self.execute_base_action(self.action_set[action], map_)
        else:
            target_unique_id = action + self.advanced_action_offset - self.base_action_dim
            target_unit = unit_dict.get(target_unique_id)
            self._action = {"type": "advanced", "real_val": [target_unit.x, target_unit.y, 0 if target_unit.map_type == "surface" else 1]}

    # For bot, sample a random action from avail action set currently.
    def set_action_for_bot(self, unit_dict, surface_map, air_map):
        if self.control_type == "bot":
            avail_acts = self.get_avail_act()
            action = np.random.choice(np.nonzero(avail_acts)[0])
            # action = 0  # for test
            self.set_action(action, unit_dict, surface_map, air_map)

    def take_action(self, unit_dict, surface_map, air_map):
        action = self._action
        reward = 0
        if action["type"] == "base":  # 已经在set_action时执行了
            return reward
            # map_ = surface_map if self.map_type == "surface" else air_map
            # self.execute_base_action(self.action_set[action], map_)
        if action["type"] == "advanced":
            x, y, z = action["real_val"]
            target_map = surface_map if z == 0 else air_map
            reward = self.execute_fire_action(unit_dict, x, y, target_map)
        return reward

    def execute_fire_action(self, unit_dict, x, y, target_map):
        target_unique_id = target_map[x, y]
        if target_unique_id == -1:
            return 0
        target_unit = unit_dict.get(target_unique_id)
        if target_unit is None:
            raise ValueError(f"Unit-{target_unique_id} not exists.")
        if target_unit.dead:
            target_map[x, y] = -1
            return 0
        # if self.camp == target_unit.camp:
        #     raise ValueError(f"Unit-{self.unique_id} fired a ally unit-{target_unique_id}")
        real_dmg = self.dmg if target_unit.hp > self.dmg else target_unit.hp
        target_unit.hp -= real_dmg
        if target_unit.dead:
            target_map[target_unit.x, target_unit.y] = -1
        return real_dmg

    @property
    def dead(self):
        return self.hp <= 0

    # observation of self
    def obs(self, units_dict):
        assert self.control_type == "agent"
        o = np.zeros(self.obs_dim, dtype=np.float32) - 1
        for unique_id, visible in enumerate(self.visible_units):
            if visible:
                unit = units_dict.get(unique_id)
                if unit.dead:
                    continue
                o[unique_id*2: unique_id*2+2] = unit.x, unit.y
                o[self.num_units*2 + unique_id] = unit.hp
        if self.obs_with_agent_id:
            o[-self.num_allies_agents:] = 0
            o[-self.num_allies_agents + self.agent_id] = 1
        return o

    def update_avail_act(self, unit_dict, surface_map, air_map):
        map_ = surface_map if self.map_type == "surface" else air_map
        avail_act = np.zeros(self.action_dim)
        avail_act[:self.base_action_dim] = self.get_avail_base_act(map_)
        avail_act[self.base_action_dim:] = self.get_avail_advanced_act(unit_dict)
        self.avail_acts = avail_act

    def get_avail_act(self):
        return self.avail_acts

    def get_avail_advanced_act(self, unit_dict):
        avail_advanced_act = np.zeros(self.advanced_action_dim)
        for unique_id, is_visible in enumerate(self.visible_units):
            if not is_visible:
                continue
            unit = unit_dict.get(unique_id)
            if unit.camp == self.camp or unit.dead:
                continue
            if self.camp == "offensive":
                avail_advanced_act[unit.unique_id - self.num_allies] = 1
            else:
                avail_advanced_act[unit.unique_id] = 1

        return avail_advanced_act

    def get_avail_base_act(self, map_):
        avail_base_act = np.ones(self.base_action_dim)
        if self.x - 1 < self.min_x or map_[self.x - 1, self.y] >= 0:
            avail_base_act[1] = 0
        if self.x + 1 > self.max_x or map_[self.x + 1, self.y] >= 0:
            avail_base_act[2] = 0
        if self.y - 1 < self.min_y or map_[self.x, self.y - 1] >= 0:
            avail_base_act[3] = 0
        if self.y + 1 > self.max_y or map_[self.x, self.y + 1] >= 0:
            avail_base_act[4] = 0
        return avail_base_act

    def update_visible_units(self, unit_dict):
        visible_units = np.zeros(self.num_units)
        self_pos = np.array([self.x, self.y])
        pos_min = np.clip(self_pos - self.sight_range, [0, 0], self.map_size)
        pos_max = np.clip(self_pos + self.sight_range, [0, 0], self.map_size)
        for unique_id, unit in unit_dict.items():
            if unit.dead:
                continue
            if np.all(pos_min <= [unit.x, unit.y]) and np.all(pos_max >= [unit.x, unit.y]):
                visible_units[unique_id] = 1
        self.visible_units = visible_units
        if self.can_receive_loc_info:
            self.merge_received_info(unit_dict)

    # 为了保证视野是最新的情况，等所有的单位视野更新(包括侦查无人机)完毕之后，再更新从侦查无人机接收到的信息
    def merge_received_info(self, unit_dict):
        for unique_id in self.received_info_from:
            scout_drone = unit_dict.get(unique_id)
            self.visible_units = merge_visible_units(self.visible_units, scout_drone.visible_units)
        self.received_info_from = []

    def set_property(self, num_rivals, num_allies, num_allies_agents, obs_with_agent_id):
        # To a "fire" action, if unit's camp is offensive, it should plus num_offensive_units to choose a defensive unit by unique_id.
        # To a "share" action, the same if unit's camp is defensive
        self.advanced_action_offset = 0 if self.camp == "defensive" else num_allies

        self.num_units = num_rivals + num_allies
        self.advanced_action_dim = num_rivals
        action_dim = self.base_action_dim + num_rivals
        self.action_dim = action_dim
        self.action_space = spaces.Discrete(action_dim)
        obs_dim = (num_allies + num_rivals) * 3
        self.obs_with_agent_id = obs_with_agent_id
        if obs_with_agent_id:
            obs_dim += num_allies_agents
        self.num_allies_agents = num_allies_agents
        self.num_allies = num_allies
        self.num_rivals = num_rivals
        self.obs_dim = obs_dim
        self.obs_space = spaces.Box(low=np.inf, high=np.inf, shape=(obs_dim, ), dtype=np.float32)

    def execute_base_action(self, action, map_):
        if action == "noop":
            return
        map_[self.x, self.y] = -1
        if action == "up":
            self.x -= 1
        elif action == "down":
            self.x += 1
        elif action == "left":
            self.y -= 1
        elif action == "right":
            self.y += 1
        else:
            raise ValueError(f"Agent: {self.agent_type} no such action: {action}")
        map_[self.x, self.y] = self.unique_id


class ScoutDrone(AgentBase):
    def __init__(self, **kwargs):
        self.map_type = "air"
        self.action_set = ['noop', 'up', 'down', 'left', 'right', 'share']
        self.base_action_dim = 5
        self.can_receive_loc_info = False
        super(ScoutDrone, self).__init__(**kwargs)

    def set_action(self, action, unit_dict, surface_map, air_map):
        avail_acts = self.get_avail_act()
        assert float(avail_acts[action]), "Action is not avail."

        if action < self.base_action_dim:
            map_ = surface_map if self.map_type == "surface" else air_map
            self._action = {"type": "base", "real_val": action}
            self.execute_base_action(self.action_set[action], map_)
        else:
            target_unique_id = action + self.advanced_action_offset - self.base_action_dim
            self._action = {"type": "advanced", "real_val": target_unique_id}

    def get_avail_advanced_act(self, unit_dict):
        avail_advanced_act = np.zeros(self.advanced_action_dim)
        for unique_id, is_visible in enumerate(self.visible_units):
            if not is_visible:
                continue
            unit = unit_dict.get(unique_id)
            if unit.camp != self.camp or unit.dead or not unit.can_receive_loc_info:
                continue
            if self.camp == "offensive":
                avail_advanced_act[unit.unique_id] = 1
            else:
                avail_advanced_act[unit.unique_id - self.num_rivals] = 1
        return avail_advanced_act

    def set_property(self, num_rivals, num_allies, num_allies_agents, obs_with_agent_id):
        self.advanced_action_offset = 0 if self.camp == "offensive" else num_rivals
        self.num_units = num_rivals + num_allies
        self.advanced_action_dim = num_allies
        action_dim = self.base_action_dim + num_allies
        self.action_dim = action_dim
        self.action_space = spaces.Discrete(action_dim)
        obs_dim = (num_allies + num_rivals) * 3
        self.obs_with_agent_id = obs_with_agent_id
        if obs_with_agent_id:
            obs_dim += num_allies_agents
        self.num_allies_agents = num_allies_agents
        self.num_allies = num_allies
        self.num_rivals = num_rivals
        self.obs_dim = obs_dim
        self.obs_space = spaces.Box(low=np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def take_action(self, unit_dict, surface_map, air_map):
        action = self._action
        reward = 0
        if action["type"] == "base":  # 已经在set_action时执行了
            return reward
            # map_ = surface_map if self.map_type == "surface" else air_map
            # self.execute_base_action(self.action_set[action], map_)
        if action["type"] == "advanced":
            received_unique_id = action["real_val"]
            received_agent = unit_dict.get(received_unique_id)
            if not received_agent.can_receive_loc_info:
                raise ValueError(f"Unit-{received_agent.unique_id} can't receive a info from scout_drone-{self.unique_id}")
            received_agent.received_info_from.append(self.unique_id)
        return reward

class MobileFort(AgentBase):
    def __init__(self, **kwargs):
        self.dmg = 10
        self.move_range = 5
        self.map_type = "surface"
        self.action_set = ['noop', 'up', 'down', 'left', 'right', 'fire']
        self.base_action_dim = 5
        self.can_receive_loc_info = True
        super(MobileFort, self).__init__(**kwargs)

    def _set_moving_boundary(self):
        # setting move range.1.在陆地范围内；2.在地图范围内；3.在移动区域内.三者交集
        # 暂时不实现第3点
        super(MobileFort, self)._set_moving_boundary()
        self.min_y = self.map_size[1] // 2

class Fighter(AgentBase):
    def __init__(self, **kwargs):
        self.dmg = 8
        self.map_type = "air"
        self.action_set = ['noop', 'up', 'down', 'left', 'right', 'fire']
        self.base_action_dim = 5
        self.can_receive_loc_info = True
        super(Fighter, self).__init__(**kwargs)


class WarShip(AgentBase):
    def __init__(self, **kwargs):
        self.dmg = 9
        self.map_type = "surface"
        self.action_set = ['noop', 'left', 'right', 'fire']
        self.base_action_dim = 3
        self.can_receive_loc_info = True
        super(WarShip, self).__init__(**kwargs)

    def get_avail_base_act(self, map_):
        avail_base_act = np.ones(3)
        if self.y-1 < self.min_y or map_[self.x, self.y-1] >= 0:
            avail_base_act[1] = 0
        if self.y+1 > self.max_y or map_[self.x, self.y+1] >= 0:
            avail_base_act[2] = 0
        return avail_base_act

    def _set_moving_boundary(self):
        super(WarShip, self)._set_moving_boundary()
        self.max_y = self.map_size[1] // 2 - 1


AGENT_CLASS_DICT = {"warship": WarShip, "fighter": Fighter, "mobile_fort": MobileFort, "scout_drone": ScoutDrone}
