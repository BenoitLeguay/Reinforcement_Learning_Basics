import numpy as np
import torch
import variable as v
from collections import deque, namedtuple
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from functools import reduce
import operator
from itertools import product
from copy import deepcopy


def get_from_dict(d, map_tuple):
    return reduce(operator.getitem, map_tuple, d)


def set_in_dict(d, map_tuple, value):
    get_from_dict(d, map_tuple[:-1])[map_tuple[-1]] = value


def moving_average(a, n=3):

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


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


class NeuralNetworkHandler:
    def __init__(self, nnh_init, optimizer_args={}, replay_buffer_args={}):
        torch.manual_seed(nnh_init['seed'])
        self.discount_factor = nnh_init["discount_factor"]
        self.eval_nn = self.init_nn(nnh_init["nn_archi"])
        self.target_nn = self.init_nn(nnh_init["nn_archi"])  # deepcopy(self.eval_nn)
        self.eval_train_delay = nnh_init["eval_train_delay"]
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_nn.parameters(), **optimizer_args)
        self.replay_buffer = ReplayBuffer(**replay_buffer_args)
        self.loss_history = list()
        self.count_step = 0.0

    @staticmethod
    def to_float_tensor(a):
        return torch.tensor(a).type(torch.FloatTensor)

    @staticmethod
    def init_nn(nn_archi):

        layers = list()

        for layer_args in nn_archi:
            layer_type = layer_args['type']
            in_features, out_features, activation = layer_args["in"], layer_args["out"], layer_args["activation"]
            layers.append(v.layer_name_to_obj[layer_type](in_features, out_features))
            if activation == 'None':
                continue
            layers.append(v.activation_name_to_obj[activation]())

        nn = torch.nn.Sequential(*layers)
        return nn

    def update_step(self, current_state, current_action, reward, next_state, done, early_stop):

        self.replay_buffer.append(current_state, current_action, reward, next_state, done)

        if self.count_step % self.eval_train_delay == 0:
            self.target_nn.load_state_dict(self.eval_nn.state_dict())
        self.count_step += 1

        if len(self.replay_buffer) == self.replay_buffer.buffer_size and not early_stop.skip_episode():
            experiences = self.replay_buffer.sample()
            self.train(experiences)

    def train(self, experiences):

        states, actions, rewards, next_states, dones = map(self.to_float_tensor, list(zip(*experiences)))

        current_values = self.eval_nn(states).gather(1, actions.long().unsqueeze(1)).squeeze()

        next_values = self.target_nn(next_states).detach()
        next_values[dones.bool()] = 0.0
        expected_values = rewards + self.discount_factor * next_values.max(1)[0]

        self.update_nn(current_values, expected_values)

    def update_nn(self, current_values, expected_values):

        loss = self.loss(current_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class ReplayBuffer:
    def __init__(self, buffer_size=1000, mini_batch_size=128, seed=42):
        random.seed(seed)
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=["state", "action", "reward", "next_state", "done"])
        self.mini_batch_size = mini_batch_size
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        return random.choices(list(self.buffer), k=self.mini_batch_size)


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
                next_action = agent.episode_init(state)
                if 'early_stop' in agent.__dict__.keys():
                    if agent.early_stop.stop_training():
                        break

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

    def append_agents(self, agents):

        assert not any(item in agents for item in self.agents), "You are trying to overwrite agents dictionary"

        self.agents.update(agents)
        self.rewards_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.time_steps_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})

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
        loss_per_agents = {'loss': {agent_name: (np.array(agent.nn_handler.loss_history) if 'nn_handler' in agent.__dict__.keys()
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


class EarlyStop:
    def __init__(self, skip_training_threshold, stop_training_threshold, episode_window_skip, episode_window_stop):
        self.skip_training_threshold = skip_training_threshold
        self.stop_training_threshold = stop_training_threshold
        self.episode_window_skip = episode_window_skip
        self.episode_window_stop = episode_window_stop
        self.episode_rewards = 0.0
        self.rewards_per_episode = deque(maxlen=episode_window_stop)
        self.skip_episode_history = list()
        self.stop_episode_history = list()

    def sum_reward(self, reward):
        self.episode_rewards += reward

    def append_sum_rewards(self):
        self.rewards_per_episode.append(self.episode_rewards)
        self.episode_rewards = 0.0

    def skip_episode(self):
        last_n_episodes = [self.rewards_per_episode[-idx] for idx in range(1, 1 + self.episode_window_skip)]
        skip_episode = np.mean(last_n_episodes) >= self.skip_training_threshold
        self.skip_episode_history.append(skip_episode)

        return skip_episode

    def stop_training(self):
        if len(self.rewards_per_episode) < self.rewards_per_episode.maxlen:
            self.stop_episode_history.append(False)
            return False
        stop_episode = np.mean(self.rewards_per_episode) >= self.stop_training_threshold
        self.stop_episode_history.append(stop_episode)

        return stop_episode
