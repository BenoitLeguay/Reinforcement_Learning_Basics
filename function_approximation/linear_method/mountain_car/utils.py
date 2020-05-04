import numpy as np
import tiles3 as tc
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def argmax(a, random_generator=None):
    if not random_generator:
        index = np.random.choice(np.flatnonzero(a == a.max()))
    else:
        index = random_generator.choice(np.flatnonzero(a == a.max()))
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
        self.rate = exploration_rate
        self.max_er = max_exploration_rate
        self.min_er = min_exploration_rate
        self.decay_er = exploration_decay_rate
        self.episode_count = 0.0
        self.constant_er = constant_exploration_rate
        self.history = list()

    def __call__(self):
        return self.rate

    def next(self):

        if self.constant_er:
            self.history.append(self.rate)
            return self.rate

        self.rate = self.min_er + ((self.max_er - self.min_er) * (np.exp(-self.decay_er * self.episode_count)))
        self.episode_count += 1.0

        self.history.append(self.rate)

        return self.rate

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


class TrainSession:
    def __init__(self, agents, env, seed):
        env.seed(seed)
        self.agents = agents
        self.env = env
        self.rewards_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}
        self.time_steps_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}

    def train(self, n_episode=500, t_max_per_episode=200, graphical=False):

        for agent_name, agent in self.agents.items():

            time_steps_per_episode = list()
            rewards_per_episode = list()

            for _ in tqdm(range(n_episode)):

                rewards = 0.0
                state = self.env.reset()
                next_action = agent.episode_init(state)

                for t in range(t_max_per_episode):
                    if graphical:
                        self.env.render()

                    state, reward, done, info = self.env.step(next_action)
                    next_action = agent.update(state, reward, done)

                    rewards += reward

                    if done:
                        break
                time_steps_per_episode.append(t)
                rewards_per_episode.append(rewards)

            self.time_steps_per_episode[agent_name] = np.concatenate([self.time_steps_per_episode[agent_name],
                                                                      np.array(time_steps_per_episode)])
            self.rewards_per_episode[agent_name] = np.concatenate([self.rewards_per_episode[agent_name],
                                                                   np.array(rewards_per_episode)])

            self.env.close()

    def append_agent(self, agents):

        assert all(item in agents for item in self.agents), "You are trying to overwrite agents dictionary"

        self.agents.update(agents)
        self.rewards_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.time_steps_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})

    def plot_results(self, moving_average_n=200):
        series_to_plot = {'rewards': self.rewards_per_episode,
                          'time_steps': self.time_steps_per_episode}

        loss_per_agents = {'loss': {agent_name: (np.array(agent.nn_handler.loss_history) if 'nn_handler' in agent.__dict__.keys()
                                                 else np.array([]))
                                    for agent_name, agent in self.agents.items()}}

        series_to_plot.update(loss_per_agents)

        for idx, (series_name, dict_series) in enumerate(series_to_plot.items()):

            figure = plt.figure(idx)
            for agent_name, series in dict_series.items():
                series_mov_avg = moving_average(series, n=moving_average_n)
                plt.plot(range(len(series_mov_avg)), series_mov_avg, label=agent_name)
            figure.suptitle(f"{series_name} per episode", fontsize=15)
            plt.ylabel(f"avg {series_name}", fontsize=10)
            plt.xlabel(f"episodes", fontsize=10)
            plt.legend()


class EligibilityTraces:
    def __init__(self, trace_decay, eligibility_shape, method):
        self.trace_decay = trace_decay
        self.traces = np.zeros(eligibility_shape)
        self.eligibility_method = method

    def __call__(self):
        return self.traces

    def episode_reset(self):
        self.traces = np.zeros(self.traces.shape)

    def update_traces(self, current_action, current_tiles):

        if self.eligibility_method == 'accumulate':
            self.traces[current_action, current_tiles] += 1
        elif self.eligibility_method == 'replace':
            self.traces[current_action, current_tiles] = 1

    def decay_traces(self, discount_factor, current_action, current_tiles):
        self.traces[current_action, current_tiles] *= discount_factor * self.trace_decay
