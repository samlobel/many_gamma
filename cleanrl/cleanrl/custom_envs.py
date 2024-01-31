# import gymnasium as gym
# from gymnasium import spaces

# import numpy as np


# class ConstantRewardEnv(gym.Env):
#     metadata = {"render_modes": ["rgb_array"]} # not really rgb.

#     def __init__(self, render_mode=None, state_size=10, action_size=4):
#         super().__init__()
#         assert isinstance(state_size, int)
#         assert isinstance(action_size, int)
#         self._state_size = state_size
#         self._action_size = action_size

#         self.observation_space = spaces.Box(-float("inf"), float("inf"), shape=(state_size,), dtype=float)
#         self.action_space = spaces.Discrete(action_size)

#     def reset(self, seed=None, options=None):
#         pass



import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


class ConstantRewardEnv(gym.Env):
    """
    Author: ChatGPT
    Custom Gymnasium environment.
    
    - state_size: Dimension of the state space.
    - action_size: Number of discrete actions.
    - The reward is always 0.
    - The next state is randomly sampled from a unit normal distribution.
    """

    def __init__(self, state_size, action_size, reward=0.0):
        super(ConstantRewardEnv, self).__init__()
        
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = gym.spaces.Discrete(action_size)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
        self._constant_reward = reward

    def step(self, action):
        # Execute one time step within the environment
        next_state = np.random.normal(0, 1, self.observation_space.shape)
        reward = self._constant_reward
        terminated = False
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info

    def reset(self, seed=None, *args, **kwargs):
        # I'm just gonna ignore the seed for now.
        # Reset the state of the environment to an initial state
        initial_state = np.random.normal(0, 1, self.observation_space.shape)
        return initial_state, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass  # Not necessary for this environment



register(
    id='ZeroRewardEnv-v0',
    entry_point='cleanrl.custom_envs:ConstantRewardEnv',
    max_episode_steps=100,
    kwargs={'state_size': 10, 'action_size': 4, 'reward': 0.0}
)

register(
    id='OneRewardEnv-v0',
    entry_point='cleanrl.custom_envs:ConstantRewardEnv',
    max_episode_steps=100,
    kwargs={'state_size': 10, 'action_size': 4, 'reward': 1.0}
)
