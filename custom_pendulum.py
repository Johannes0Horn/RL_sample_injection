import gym
import numpy as np


class CustomPendulum(gym.Wrapper):

    def __init__(self, env, _seed=0):
        super().__init__(env)
        # overwrite seed
        self.seed(_seed)

    def reset(self, starting_state=None):
        if starting_state is None:
            high = np.array([np.pi, 1])
            self.env.state = self.env.np_random.uniform(low=-high, high=high)
            self.env.last_u = None
            return self.env._get_obs()
        else:
            self.env.state = starting_state
            self.env.last_u = None
            return self.env._get_obs()

    def get_state(self):
        return self.env.state

    def set_state(self, state):
        self.env.state = state
