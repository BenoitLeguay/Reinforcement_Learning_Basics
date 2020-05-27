import utils
import numpy as np
import torch


class REINFORCEAgent:
    def __init__(self, agent_init):
        torch.manual_seed(agent_init["seed"])
        self.discount_factor = agent_init["discount_factor"]
        self.num_action = agent_init["num_action"]
        self.random_generator = np.random.RandomState(seed=agent_init["seed"])
        self.nn = utils.init_nn(agent_init["nn_archi"])
        self.optimizer = torch.optim.Adam(self.nn.parameters(), **agent_init["optimizer"])
        self.loss_history = list()

    def train(self, episode):
        discounted_reward = 0.0
        self.optimizer.zero_grad()

        for prob_action_taken, reward in reversed(episode):
            discounted_reward = reward + self.discount_factor * discounted_reward
            loss = -torch.log(prob_action_taken) * discounted_reward
            self.loss_history.append(loss.item())
            loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        probs = self.predict(state)
        action = self.random_generator.choice(self.num_action, p=probs.data.numpy())
        return action, probs[action]

    def predict(self, state):
        state = utils.to_tensor(state)
        return self.nn(state)


class OffPolicyAgent:
    pass
