import numpy as np


def _cast(x):
    return x.transpose(2, 0, 1, 3)


class RecReplayBuffer:
    def __init__(self, policy_info, policy_agents, buffer_size, episode_length, use_avail_acts,
                 use_reward_normalization=False):
        self.policy_info = policy_info

        self.policy_buffers = {p_id: RecBuffer(buffer_size,
                                               episode_length,
                                               len(policy_agents[p_id]),
                                               self.policy_info[p_id]['obs_space'],
                                               self.policy_info[p_id]['share_obs_space'],
                                               self.policy_info[p_id]['act_space'],
                                               use_avail_acts,
                                               use_reward_normalization)
                               for p_id in self.policy_info.keys()}

    def __len__(self):
        return self.policy_buffers['policy_0'].filled_i

    def insert(self, num_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts):
        idx_range = None
        for p_id in self.policy_info.keys():
            idx_range = self.policy_buffers[p_id].insert(num_episodes, np.array(obs[p_id]),
                                                         np.array(share_obs[p_id]), np.array(acts[p_id]),
                                                         np.array(rewards[p_id]), np.array(dones[p_id]),
                                                         np.array(dones_env[p_id]), np.array(avail_acts[p_id]))
        return idx_range

    def sample(self, batch_size):
        inds = np.random.choice(self.__len__(), batch_size)
        obs, share_obs, acts, rewards, dones, dones_env, avail_acts = {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            obs[p_id], share_obs[p_id], acts[p_id], rewards[p_id], dones[p_id], dones_env[p_id], avail_acts[p_id] = \
                self.policy_buffers[p_id].sample_inds(inds)

        return obs, share_obs, acts, rewards, dones, dones_env, avail_acts


class RecBuffer:
    def __init__(self, buffer_size, episode_length, num_agents, obs_dim, share_obs_dim, act_dim,
                 use_avail_acts, use_reward_normalization=False):
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.num_agents = num_agents
        # self.use_same_share_obs = use_same_share_obs # always True in m2ale
        self.use_avail_acts = use_avail_acts
        self.use_reward_normalization = use_reward_normalization
        self.filled_i = 0
        self.current_i = 0

        self.obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, obs_dim), dtype=np.float32)
        self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, share_obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.episode_length, self.buffer_size, self.num_agents, act_dim), dtype=np.float32)
        self.avail_acts = np.ones((self.episode_length + 1, self.buffer_size, self.num_agents, act_dim),
                                  dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones((self.episode_length, self.buffer_size, 1), dtype=np.float32)

    def __len__(self):
        return self.filled_i

    def insert(self, num_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts=None):
        # TODO Do not must fulfill
        episode_length = acts.shape[0]
        assert episode_length == self.episode_length, "different dimension!"

        if self.current_i + num_episodes <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_episodes)
        else:
            num_left_episodes = self.current_i + num_episodes - self.buffer_size
            idx_range = np.concatenate((np.arange(self.current_i, self.buffer_size), np.arange(num_left_episodes)))

        # remove agent dimension since all agents share centralized observation
        share_obs = share_obs[:, :, 0]  # all agents use the same state

        self.obs[:, idx_range] = obs.copy()
        self.share_obs[:, idx_range] = share_obs.copy()
        self.acts[:, idx_range] = acts.copy()
        self.avail_acts[:, idx_range] = avail_acts.copy()
        self.rewards[:, idx_range] = rewards.copy()
        self.dones[:, idx_range] = dones.copy()
        self.dones_env[:, idx_range] = dones_env.copy()

        self.current_i = idx_range[-1] + 1
        self.filled_i = min(self.filled_i + len(idx_range), self.buffer_size)

        return idx_range

    def sample_inds(self, sample_inds):
        # obs: [step, episode, agent, dim] => [agent, step, episode, dim]
        obs = _cast(self.obs[:, sample_inds])
        acts = _cast(self.acts[:, sample_inds])
        if self.use_reward_normalization:
            # mean std
            # [length, envs, agents, 1]
            # [length, envs, 1]
            all_dones_env = np.tile(np.expand_dims(
                self.dones_env[:, :self.filled_i], -1), (1, 1, self.num_agents, 1))
            first_step_dones_env = np.zeros((1, self.filled_i, self.num_agents, 1))
            curr_dones_env = np.concatenate((first_step_dones_env, all_dones_env[:self.episode_length - 1]))
            temp_rewards = self.rewards[:, :self.filled_i].copy()
            temp_rewards[curr_dones_env == 1.0] = np.nan

            mean_reward = np.nanmean(temp_rewards)
            std_reward = np.nanstd(temp_rewards)
            rewards = _cast((self.rewards[:, sample_inds] - mean_reward) / std_reward)
        else:
            rewards = _cast(self.rewards[:, sample_inds])

        share_obs = self.share_obs[:, sample_inds]
        dones = _cast(self.dones[:, sample_inds])
        dones_env = self.dones_env[:, sample_inds]
        avail_acts = _cast(self.avail_acts[:, sample_inds])

        return obs, share_obs, acts, rewards, dones, dones_env, avail_acts
