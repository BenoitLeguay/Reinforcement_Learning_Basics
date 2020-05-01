import numpy as np
import torch
import variable as v
from collections import deque, namedtuple
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def argmax(a):
    index = np.random.choice(np.flatnonzero(a == a.max()))
    return index


class NeuralNetworkHandler:
    def __init__(self, nnh_init, optimizer_args={}, replay_buffer_args={}):
        torch.manual_seed(nnh_init['seed'])
        self.test = list()
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

    def update_step(self, current_state, current_action, reward, next_state, done):

        self.replay_buffer.append(current_state, current_action, reward, next_state, done)

        if self.count_step % self.eval_train_delay == 0:
            self.target_nn.load_state_dict(self.eval_nn.state_dict())
        self.count_step += 1

        if len(self.replay_buffer) == self.replay_buffer.buffer_size:
            experiences = self.replay_buffer.sample()
            self.train(experiences)
            self.test.append(1.0)
        else:
            self.test.append(0.0)

    def train(self, experiences):

        states, actions, rewards, next_states, dones = map(self.to_float_tensor, list(zip(*experiences)))

        current_values = self.eval_nn(states).gather(1, actions.long().unsqueeze(1)).squeeze()

        next_values = self.target_nn(next_states).detach()
        next_values[dones.bool()] = 0.0
        expected_values = rewards + self.discount_factor * next_values.max(1)[0]

        self.update_nn(current_values, expected_values)

    def update_nn(self, current_values, expected_values):

        loss = self.loss(current_values, expected_values)
        #  loss.requires_grad = True
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


