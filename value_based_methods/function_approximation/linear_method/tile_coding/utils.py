import numpy as np
import tiles3 as tc
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from functools import reduce
import operator


def get_from_dict(d, map_tuple):
    return reduce(operator.getitem, map_tuple, d)


def set_in_dict(d, map_tuple, value):
    get_from_dict(d, map_tuple[:-1])[map_tuple[-1]] = value


def rolling_window(a, window=3):

    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


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
        plt.style.use('ggplot')
        self.agents = agents
        self.env = env
        self.rewards_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}
        self.time_steps_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}
        self.line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
        self.num_lines_style = len(self.line_styles)
        self.cm = plt.get_cmap('tab10')
        self.max_diff_colors = 8

    def train(self, n_episode=500, t_max_per_episode=200, graphical=False, agent_subset=None):

        if agent_subset:
            agents = {agent_name: self.agents[agent_name] for agent_name in agent_subset}
        else:
            agents = self.agents

        for agent_name, agent in agents.items():

            time_steps_per_episode = list()
            rewards_per_episode = list()

            for _ in tqdm(range(n_episode)):

                rewards = 0.0
                state = self.env.reset()
                next_action = agent.episode_init(state)

                for t in range(t_max_per_episode):
                    if graphical:
                        self.env.render()

                    state, reward, done, info = self.env.step(next_action)  # problem when the for loop end, while done is not True (agent_end not called)
                    next_action = agent.update(state, reward, done)

                    rewards += reward

                    if done:
                        break

                if 'early_stop' in agent.__dict__.keys():
                    agent.early_stop.append_sum_rewards(rewards)
                    if agent.early_stop.stop_training():
                        break

                time_steps_per_episode.append(t)
                rewards_per_episode.append(rewards)
                agent.exploration_handler.next()

            self.time_steps_per_episode[agent_name] = np.concatenate([self.time_steps_per_episode[agent_name],
                                                                      np.array(time_steps_per_episode)])
            self.rewards_per_episode[agent_name] = np.concatenate([self.rewards_per_episode[agent_name],
                                                                   np.array(rewards_per_episode)])

            self.env.close()

    def append_agents(self, agents, overwrite=False):

        assert not any(item in agents for item in self.agents) or overwrite, "You are trying to overwrite agents dictionary"
        agent_names = list(agents.keys())

        self.agents.update(agents)
        self.rewards_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.time_steps_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})

        return agent_names

    def pop_agents(self, agents):
        valid_agent_name = set(agents).intersection(self.agents.keys())
        for agent_name in valid_agent_name:
            self.agents.pop(agent_name)

    def parameter_grid_append(self, agent_object, base_agent_init, parameters_dict):

        agents = {}
        parameter_grid = list(dict(zip(parameters_dict, x)) for x in product(*parameters_dict.values()))
        for parameters_dict in parameter_grid:
            agent_init_tmp = deepcopy(base_agent_init)
            agent_name = ""
            for name, value in parameters_dict.items():
                set_in_dict(agent_init_tmp, name, value)
                agent_name += f"{'_'.join(name)}:{value};"

            agents.update({agent_name: agent_object(agent_init_tmp)})
            self.rewards_per_episode.update({agent_name: np.array([])})
            self.time_steps_per_episode.update({agent_name: np.array([])})

        self.agents.update(agents)

        return list(agents.keys())

    def plot_results(self, window=200, agent_subset=None, std=True):

        if not agent_subset:
            agent_subset = self.agents.keys()

        series_to_plot = {'rewards': {agent_name: self.rewards_per_episode[agent_name] for agent_name in agent_subset},
                          'time_steps': {agent_name: self.time_steps_per_episode[agent_name] for agent_name in agent_subset}}

        agents_to_plot = {agent_name: self.agents[agent_name] for agent_name in agent_subset}
        loss_per_agents = {'loss': {agent_name: (np.array(agent.dqn_handler.loss_history) if 'dqn_handler' in agent.__dict__.keys()
                                                 else np.array([]))
                                    for agent_name, agent
                                    in agents_to_plot.items()}}

        series_to_plot.update(loss_per_agents)

        fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(10, 20), facecolor='w', edgecolor='k')
        axs = axs.ravel()

        for idx, (series_name, dict_series) in enumerate(series_to_plot.items()):
            for jdx, (agent_name, series) in enumerate(dict_series.items()):
                if series.size == 0:
                    axs[idx].plot([0.0], [0.0], label=agent_name)
                    continue

                cm_idx = jdx % self.max_diff_colors # jdx // self.num_lines_style * float(self.num_lines_style) / self.max_diff_colors
                ls_idx = min(jdx // self.max_diff_colors, self.num_lines_style)  # jdx % self.num_lines_style

                series_mvg = rolling_window(series, window=window)
                series_mvg_avg = np.mean(series_mvg, axis=1)

                lines = axs[idx].plot(range(len(series_mvg_avg)), series_mvg_avg, label=agent_name)

                lines[0].set_color(self.cm(cm_idx))
                lines[0].set_linestyle(self.line_styles[ls_idx])

                if std:
                    series_mvg_std = np.std(series_mvg, axis=1)
                    area = axs[idx].fill_between(range(len(series_mvg_avg)), series_mvg_avg - series_mvg_std,
                                                 series_mvg_avg + series_mvg_std, alpha=0.15)
                    area.set_color(self.cm(cm_idx))
                    area.set_linestyle(self.line_styles[ls_idx])

            box = axs[idx].get_position()
            axs[idx].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axs[idx].set_title(f"{series_name} per episode", fontsize=15)
            axs[idx].set_ylabel(f"avg {series_name}", fontsize=10)
            axs[idx].set_xlabel(f"episodes", fontsize=10)
            axs[idx].legend(loc='center left', bbox_to_anchor=(1, 0.5))


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

    def decay_traces(self, discount_factor):
        self.traces *= discount_factor * self.trace_decay


class TrueEligibilityTraces:
    def __init__(self, trace_decay, eligibility_shape):
        self.trace_decay = trace_decay
        self.traces = np.zeros(eligibility_shape)

    def __call__(self):
        return self.traces

    def episode_reset(self):
        self.traces = np.zeros(self.traces.shape)

    def update_traces(self, learning_rate, discount_factor, current_action, current_tiles):
        self.traces *= discount_factor * self.trace_decay
        self.traces[current_action, current_tiles] += 1 - learning_rate * discount_factor * self.trace_decay * np.sum(self.traces[current_action, current_tiles])
