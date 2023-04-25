import numpy as np


class EpisodeRunner:

    def __init__(self, env, eval_env, policies, agents_policies, policies_agents, num_agents, hidden_size):
        self.env = env
        self.eval_env = eval_env
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        # policies
        self.num_agents = num_agents
        self.num_envs = 1  # episode runner fix to 1
        self.hidden_size = hidden_size

        self.policy_agents = policies_agents  # policies => agents, 1=>more
        self.policies = policies  # all policies
        self.agents_policies = agents_policies  # agents => policies
        self.policy_ids = self.policy_agents.keys()  # id of all policies

        # self.policy_info = config["policy_info"]
        # self.policy_ids = sorted(list(self.policy_info.keys()))
        # self.policy_mapping_fn = config["policy_mapping_fn"]
        # self.agent_ids = [i for i in range(self.num_agents)]
        # self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}
        # # map policy id to agent ids controlled by that policy
        # self.policy_agents = {
        #     policy_id: sorted([agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id])
        #     for policy_id in self.policies.keys()}

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def close_env(self):
        self.env.close()

    def reset(self):
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        env_info = {}
        env = self.eval_env if test_mode else self.env

        self.reset()
        obs = env.get_obs()
        # TODO check if is initialize with zeros
        rnn_states = np.zeros((self.num_agents, self.num_envs, self.hidden_size))

        # episode trajectory
        episode_obs = {p_id: np.zeros((self.episode_limit + 1, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_state = {p_id: np.zeros((self.episode_limit + 1, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].state), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id: np.zeros((self.episode_limit, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].act_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id: np.zeros((self.episode_limit, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id: np.ones((self.episode_limit, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id: np.ones((self.episode_limit, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}
        last_acts = {p_id: np.zeros((self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].act_dim), dtype=np.float32) for p_id in self.policy_ids}
        rnn_states_batch = {p_id: np.zeros((self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].act_dim), dtype=np.float32) for p_id in self.policy_ids}

        terminated = False
        episode_return = 0
        while not terminated and self.t < self.episode_limit:
            state = self.env.get_state()
            obs = self.env.get_obs()
            avail_actions = self.env.get_avail_actions()

            for agent_id, p_id in self.agents_policies.items():
                policy = self.policies[p_id]
                # [30] 也许需要叠加一下才能使用网络
                agent_obs = np.stack(obs[agent_id])
                state = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(self.num_envs, -1).astype(np.float32)



            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, timestep_to_load=self.timestep_to_load)

            reward, terminated, env_info = self.env.step(actions[0])
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, timestep_to_load=self.timestep_to_load)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
