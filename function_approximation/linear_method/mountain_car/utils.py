import numpy as np
import tiles3 as tc
from copy import deepcopy


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def argmax(a):

    index = np.random.choice(np.flatnonzero(a == a.max()))
    return index


class TileCoder:
    def __init__(self, num_tiles, num_tilings, hash_size, position_boundaries, velocity_boundaries):
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.hash_size = hash_size
        self.iht = tc.IHT(hash_size)

        self.position_scale = self.num_tiles / (position_boundaries[1] - position_boundaries[0])
        self.velocity_scale = self.num_tiles / (velocity_boundaries[1] - velocity_boundaries[0])

    def index_feature_to_mask_feature(self, tiles):
        feature = np.zeros(self.hash_size)
        feature[tiles] = 1.0

        return feature

    def get_active_tiles(self, state):
        position, velocity = state
        state_scaled = [position * self.position_scale, velocity * self.velocity_scale]

        active_tiles = tc.tiles(self.iht, self.num_tilings, state_scaled)

        return np.array(active_tiles)


class ExplorationRateDecay:
    def __init__(self, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, constant_exploration_rate):
        self.er = exploration_rate
        self.max_er = max_exploration_rate
        self.min_er = min_exploration_rate
        self.decay_er = exploration_decay_rate
        self.episode_count = 0.0
        self.constant_er = constant_exploration_rate
        self.history = list()

    def next(self):

        if self.constant_er:
            self.history.append(self.er)
            return self.er

        self.er = self.min_er + ((self.max_er - self.min_er) * (np.exp(-self.decay_er * self.episode_count)))
        self.episode_count += 1.0

        self.history.append(self.er)

        return self.er

    def reset(self):
        self.episode_count = 0.0


class AdaptiveLearningRate:
    """
    https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5092/5494
    """
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.history = list()

    def next(self, current_feature, next_feature, eligibility, gamma):
        print(f"current_feature: {current_feature.shape}, next_feature: {next_feature.shape}, eligibility: {eligibility.shape}")
        possible_next_epsilon = 1/(np.abs(np.dot(eligibility, gamma * next_feature - current_feature)))
        self.epsilon = min(self.epsilon, possible_next_epsilon)
        self.history.append(deepcopy(self.epsilon))

        return self.epsilon
