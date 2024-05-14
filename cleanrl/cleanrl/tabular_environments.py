import gymnasium as gym
import numpy as np



def sample_random_simplex_vector(n):
    # Returns a random vector on the n-dimensional simplex.
    # Reference: https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    x = np.random.exponential(size=n)
    return x / np.sum(x)


def sample_transition_function(states, actions):
    # Returns a random vector on the n-dimensional simplex for each state-action.
    # Reference: https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    x = np.random.exponential(size=(states, actions, states))
    return x / np.sum(x, axis=-1, keepdims=True)

def q_value_iteration(env, gamma=0.99, theta=1e-8):
    # copilot, not confirmed. Also, not Q values yet, so I should fix that.
    # V = np.zeros(env.observation_space.shape)
    obs_space_shape = env.observation_space.shape
    assert len(obs_space_shape) == 1
    num_states = obs_space_shape[0]
    Q_shape = (num_states, int(env.action_space.n))
    # print(Q_shape)
    # 
    Q = np.zeros(Q_shape)
    # Q = np.zeros((env.observation_space.shape, env.action_space.shape))
    iter_num = 0
    while True:
        delta = 0
        for s in range(env.num_states):
            for a in range(env.num_actions):
                r = env.reward_matrix[s, a]
                probabilities = env.transition_tensor[s, a]
                qsa = r + gamma * np.sum([p * np.max(Q[s_, :]) for p, s_ in zip(probabilities, range(env.num_states))])
                delta = max(delta, abs(qsa - Q[s,a]))
                Q[s, a] = qsa
        if iter_num % 20 == 0:
            print("Iteration: ", iter_num, "Delta: ", delta)
        if delta < theta:
            break
        iter_num += 1
    policy = np.argmax(Q, axis=-1)
    return Q, policy

def get_q_values_for_policy(env, policy, gamma=0.99, theta=1e-8):
    # policy is array of integers that represent action at that state. Goal should be compute V, and then return
    # Q through R + PV.
    V = np.zeros(env.observation_space.shape)
    iter_num = 0
    while True:
        delta = 0
        for s in range(env.num_states):
            a = policy[s]
            r = env.reward_matrix[s, a]
            probabilities = env.transition_tensor[s, a]
            vs = r + gamma * np.sum([p * V[s_] for p, s_ in zip(probabilities, range(env.num_states))])
            delta = max(delta, abs(vs - V[s]))
            V[s] = vs
        if iter_num % 20 == 0:
            print("Iteration: ", iter_num, "Delta: ", delta)
        if delta < theta:
            break
        iter_num += 1
    Q = np.zeros((env.num_states, env.num_actions))
    for s in range(env.num_states):
        for a in range(env.num_actions):
            r = env.reward_matrix[s, a]
            probabilities = env.transition_tensor[s, a]
            Q[s, a] = r + gamma * np.sum([p * V[s_] for p, s_ in zip(probabilities, range(env.num_states))])
    return Q


class TabularEnv(gym.Env):
    # No done for now.
    def __init__(self, num_states, num_actions, transition_tensor, reward_matrix, initial_distribution):
        self.num_states = num_states
        self.num_actions = num_actions
        self.state = 0
        self.action_space = gym.spaces.Discrete(num_actions)
        # self.observation_space = gym.spaces.Discrete(num_states)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(num_states,), dtype=np.float32) # one-hot, sadly
        self.transition_tensor = transition_tensor
        self.initial_distribution = initial_distribution
        self.reward_matrix = reward_matrix
        assert self.transition_tensor.shape == (num_states, num_actions, num_states), self.transition_tensor.shape
        assert self.reward_matrix.shape == (num_states, num_actions)
        assert self.initial_distribution.shape == (num_states,), self.initial_distribution.shape
        assert np.allclose(np.sum(self.transition_tensor, axis=-1), 1)
        assert np.allclose(np.sum(self.initial_distribution), 1)
        assert np.all(self.reward_matrix >= 0)
        assert np.all(self.reward_matrix <= 1)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.state = np.random.choice(self.num_states, p=self.initial_distribution)
        return self._one_hot(self.state), {'state_integer': self.state}

    def _one_hot(self, state):
        state_array = np.zeros(self.num_states, dtype=np.float32)
        state_array[state] = 1
        return state_array

    def step(self, action):
        next_state = np.random.choice(self.num_states, p=self.transition_tensor[self.state, action])
        reward = self.reward_matrix[self.state, action]
        terminated = False
        truncated = False # Assume timelimit wrapper overwrites this
        self.state = next_state
        return self._one_hot(next_state), reward, terminated, truncated, {'state_integer': self.state}
    
    def get_optimal_q_values_and_policy(self, gamma=0.99):
        return q_value_iteration(self, gamma=gamma)

    def get_manygamma_values(self, gammas, gamma_to_choose=None):
        if gamma_to_choose is None:
            gamma_to_choose = gammas[-1]
        _, pi = self.get_optimal_q_values_and_policy(gamma=gamma_to_choose)
        all_Qs = []
        for gamma in gammas:
            Q = get_q_values_for_policy(self, pi, gamma=gamma)
            all_Qs.append(Q[:,None,:])
        return np.concatenate(all_Qs, axis=1)
        


class RandomTabularEnv(TabularEnv):
    # amount_noise_prob is just for compatibility
    def __init__(self, num_states=10, num_actions=2, amount_noise_prob=0.1):
        transition_tensor = sample_transition_function(num_states, num_actions)
        reward_matrix = np.random.uniform(size=(num_states, num_actions))
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

    # def reset(self):
    #     self.state = 0
    #     return self.state
    
    # def step(self, action):
    #     next_state = np.random.choice(self.num_states, p=self.transition_tensor[self.state, action])
    #     reward = 1 if next_state == self.num_states - 1 else 0
    #     done = next_state == self.num_states - 1
    #     self.state = next_state
    #     return next_state, reward, done, {}

class SparseRewardRandomTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2, amount_noise_prob=0.1):
        transition_tensor = sample_transition_function(num_states, num_actions)
        reward_matrix = np.zeros((num_states, num_actions))
        reward_matrix[-1, -1] = 1
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

class SparseRewardSparsishTransitionTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2, amount_noise_prob=0.1):
        transition_tensor = sample_transition_function(num_states, num_actions)
        transition_tensor = transition_tensor**4
        transition_tensor = transition_tensor / np.sum(transition_tensor, axis=-1, keepdims=True)
        reward_matrix = np.zeros((num_states, num_actions))
        reward_matrix[-1, -1] = 1
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

class SparseRewardAbsorbingStateTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2, amount_noise_prob=0.1):
        transition_tensor = sample_transition_function(num_states, num_actions)
        transition_tensor[0] = 0
        transition_tensor[0,:,0] = 1
        reward_matrix = np.zeros((num_states, num_actions))
        reward_matrix[-1, -1] = 1
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

class NoisyRingTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2, amount_noise_prob=0.1):
        assert num_states > 2 and num_actions == 2
        transition_tensor = np.ones((num_states, num_actions, num_states), dtype=np.float32) * amount_noise_prob / (num_states - 1)
        for i in range(num_states):
            # print(i, 0, (num_states + i + 1 ) % num_states)
            # print(i, 1, (num_states + i - 1 ) % num_states)
            transition_tensor[i, 0, (num_states + i + 1 ) % num_states] = (1 - amount_noise_prob) 
            transition_tensor[i, 1, (num_states + i - 1 ) % num_states] = (1 - amount_noise_prob) 

        # transition_tensor = sample_transition_function(num_states, num_actions)
        # transition_tensor[0] = 0
        # transition_tensor[0,:,0] = 1
        reward_matrix = np.zeros((num_states, num_actions))
        reward_matrix[-1, :] = 1
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

class AllOneRewardTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2, amount_noise_prob=0.1):
        transition_tensor = sample_transition_function(num_states, num_actions)
        reward_matrix = np.ones((num_states, num_actions), dtype=np.float32)
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

class AllZeroRewardTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2, amount_noise_prob=0.1):
        transition_tensor = sample_transition_function(num_states, num_actions)
        reward_matrix = np.zeros((num_states, num_actions), dtype=np.float32)
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

gym.register("RandomTabularEnv-v0", entry_point=RandomTabularEnv, max_episode_steps=200)
gym.register("SparseRewardRandomTabularEnv-v0", entry_point=SparseRewardRandomTabularEnv, max_episode_steps=200)
gym.register("SparseRewardAbsorbingStateTabularEnv-v0", entry_point=SparseRewardAbsorbingStateTabularEnv, max_episode_steps=200)
gym.register("NoisyRingTabularEnv-v0", entry_point=NoisyRingTabularEnv, max_episode_steps=200)
gym.register("AllOneRewardTabularEnv-v0", entry_point=AllOneRewardTabularEnv, max_episode_steps=200)


if __name__ == "__main__":
    # env = RandomTabularEnv()
    # env = SparseRewardRandomTabularEnv()
    # env = SparseRewardAbsorbingStateTabularEnv()
    env = NoisyRingTabularEnv(num_states=10)
    Q, policy = q_value_iteration(env, gamma=0.5)
    print(Q)
    print(policy)
    