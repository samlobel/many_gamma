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

def value_iteration(env, gamma=0.99, theta=1e-8):
    # copilot, not confirmed. Also, not Q values yet, so I should fix that.
    # V = np.zeros(env.observation_space.shape)
    Q_shape = (int(env.observation_space.n), int(env.action_space.n))
    # print(Q_shape)
    # import ipdb; ipdb.set_trace()
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
        print("Iteration: ", iter_num, "Delta: ", delta)
        if delta < theta:
            break
        iter_num += 1
    policy = np.argmax(Q, axis=-1)
    return Q, policy

class TabularEnv(gym.Env):
    # No done for now.
    def __init__(self, num_states, num_actions, transition_tensor, reward_matrix, initial_distribution):
        self.num_states = num_states
        self.num_actions = num_actions
        self.state = 0
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Discrete(num_states)
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
    
    def reset(self):
        self.state = np.random.choice(self.num_states, p=self.initial_distribution)
        return self.state

    def step(self, action):
        next_state = np.random.choice(self.num_states, p=self.transition_tensor[self.state, action])
        reward = self.reward_matrix[self.state, action]
        done = False
        self.state = next_state
        return next_state, reward, done, {}


class RandomTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2):
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
    def __init__(self, num_states=10, num_actions=2):
        transition_tensor = sample_transition_function(num_states, num_actions)
        reward_matrix = np.zeros((num_states, num_actions))
        reward_matrix[-1, -1] = 1
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

class SparseRewardSparsishTransitionTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2):
        transition_tensor = sample_transition_function(num_states, num_actions)
        transition_tensor = transition_tensor**4
        transition_tensor = transition_tensor / np.sum(transition_tensor, axis=-1, keepdims=True)
        reward_matrix = np.zeros((num_states, num_actions))
        reward_matrix[-1, -1] = 1
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)

class SparseRewardAbsorbingStateTabularEnv(TabularEnv):
    def __init__(self, num_states=10, num_actions=2):
        transition_tensor = sample_transition_function(num_states, num_actions)
        transition_tensor[0] = 0
        transition_tensor[0,:,0] = 1
        reward_matrix = np.zeros((num_states, num_actions))
        reward_matrix[-1, -1] = 1
        initial_distribution = sample_random_simplex_vector(num_states)
        super().__init__(num_states, num_actions, transition_tensor, reward_matrix, initial_distribution)


if __name__ == "__main__":
    # env = RandomTabularEnv()
    # env = SparseRewardRandomTabularEnv()
    env = SparseRewardAbsorbingStateTabularEnv()
    Q, policy = value_iteration(env, gamma=0.99)
    print(Q)
    print(policy)
    