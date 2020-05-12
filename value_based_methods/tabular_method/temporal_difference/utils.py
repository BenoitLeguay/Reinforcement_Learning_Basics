import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import reduce
import operator
from itertools import product


def get_from_dict(d, map_tuple):
    return reduce(operator.getitem, map_tuple, d)


def set_in_dict(d, map_tuple, value):
    get_from_dict(d, map_tuple[:-1])[map_tuple[-1]] = value


def rolling_window(a, window=3):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window - 1
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


class EligibilityTraces:
    def __init__(self, trace_decay, eligibility_shape, method):
        self.trace_decay = trace_decay
        self.traces = np.zeros(eligibility_shape)
        self.eligibility_method = method

    def __call__(self):
        return self.traces

    def episode_reset(self):
        self.traces = np.zeros(self.traces.shape)

    def update_traces(self, current_action, current_state):

        if self.eligibility_method == 'accumulate':
            self.traces[current_action, current_state] += 1
        elif self.eligibility_method == 'replace':
            self.traces[current_action, current_state] = 1

    def decay_traces(self, discount_factor):
        self.traces *= discount_factor * self.trace_decay


class TrainSession:
    def __init__(self, agents, env, seed):
        env.seed(seed)
        plt.style.use('ggplot')
        self.agents = agents
        self.env = env
        self.rewards_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}
        self.time_steps_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}

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
                state = agent.state_to_index(state)
                next_action = agent.episode_init(state)

                for t in range(t_max_per_episode):
                    if graphical:
                        self.env.render()

                    state, reward, done, info = self.env.step(next_action)
                    state = agent.state_to_index(state)
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

    def append_agents(self, agents):

        assert not any(item in agents for item in self.agents), "You are trying to overwrite agents dictionary"

        self.agents.update(agents)
        self.rewards_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.time_steps_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})

        return list(agents.keys())

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

    def plot_results(self, window=200, agent_subset=None):

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
            for agent_name, series in dict_series.items():
                if series.size == 0:
                    axs[idx].plot([0.0], [0.0], label=agent_name)
                    continue

                series_mvg = rolling_window(series, window=window)
                series_mvg_avg = np.mean(series_mvg, axis=1)
                series_mvg_std = np.std(series_mvg, axis=1)

                axs[idx].plot(range(len(series_mvg_avg)), series_mvg_avg, label=agent_name)
                axs[idx].fill_between(range(len(series_mvg_avg)), series_mvg_avg - series_mvg_std,
                                      series_mvg_avg + series_mvg_std, alpha=0.15)

            box = axs[idx].get_position()
            axs[idx].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axs[idx].set_title(f"{series_name} per episode", fontsize=15)
            axs[idx].set_ylabel(f"avg {series_name}", fontsize=10)
            axs[idx].set_xlabel(f"episodes", fontsize=10)
            axs[idx].legend(loc='center left', bbox_to_anchor=(1, 0.5))
