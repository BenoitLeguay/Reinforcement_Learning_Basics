import numpy as np
import torch
import variable as v


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def argmax(a):
    index = np.random.choice(np.flatnonzero(a == a.max()))
    return index


class NeuralNetworkHandler:
    def __init__(self, nn_archi, optimizer_args):
        self.nn = self.init_nn(nn_archi)
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), *optimizer_args)
        self.loss_history = list()

    @staticmethod
    def state_to_tensor(state):
        return torch.tensor(state).type(torch.FloatTensor)

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

    def update_nn(self, current_values, expected_values):

        loss = self.loss(current_values, expected_values)
        loss.requires_grad = True
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class ExplorationRateDecay:
    def __init__(self, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, constant_exploration_rate):
        self.rate = exploration_rate
        self.max_er = max_exploration_rate
        self.min_er = min_exploration_rate
        self.decay_er = exploration_decay_rate
        self.episode_count = 0.0
        self.constant_er = constant_exploration_rate
        self.history = list()

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