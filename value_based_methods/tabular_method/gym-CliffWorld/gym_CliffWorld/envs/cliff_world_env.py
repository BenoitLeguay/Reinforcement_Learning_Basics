import gym
from copy import deepcopy
import sys
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

grid = ["SXXXXXXXXXXG",
        "____________",
        "____________",
        "____________"]


class CliffWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = [0, 0]
        self.goal = [0, 11]
        self.current_state = None
        self.reward_obs_term = None
        self.action_space = 4
        self.state_space = self.rows * self.cols
        self.grid = grid

    def step(self, action):
        new_state = deepcopy(self.current_state)

        if action == 0: #right
            new_state[1] = min(new_state[1]+1, self.cols-1)
        elif action == 1: #down
            new_state[0] = max(new_state[0]-1, 0)
        elif action == 2: #left
            new_state[1] = max(new_state[1]-1, 0)
        elif action == 3: #up
            new_state[0] = min(new_state[0]+1, self.rows-1)
        else:
            raise Exception("Invalid action.")
        self.current_state = new_state

        reward = -1.0
        is_terminal = False
        if self.current_state[0] == 0 and self.current_state[1] > 0 and not self.current_state == self.start:
            if self.current_state[1] < self.cols - 1:
                reward = -100.0
                is_terminal = True
            else:
                reward = 100.0
                is_terminal = True

        self.reward_obs_term = (self.observation(self.current_state), reward, is_terminal, {})

        return self.reward_obs_term

    def reset(self):

        self.current_state = self.start  # An empty NumPy array

        self.reward_obs_term = (0.0, self.observation(self.current_state), False)

        return self.reward_obs_term[1]

    def render(self, mode='human', close=False):

        outfile = sys.stdout

        tmp_grid = self.grid.copy()

        tmp_grid_array = np.asarray(tmp_grid, dtype='c')
        tmp_grid_array[self.current_state[0], self.current_state[1]] = 'T'

        tmp_grid_bytes = [[c.decode('utf-8') for c in line] for line in tmp_grid_array]
        tmp_grid_str = [''.join(row) for row in tmp_grid_bytes]
        outfile.write("\n".join(''.join(line) for line in tmp_grid_str) + "\n")

    def observation(self, state):
        return state[0] * self.cols + state[1]

