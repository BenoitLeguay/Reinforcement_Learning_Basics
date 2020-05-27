import numpy as np
import utils
import torch
import variable as v
from torch.distributions import Categorical
from copy import deepcopy
from collections import namedtuple
import abc


class PPOAgent:
    """

    """
    def __init__(self, agent_init):
        torch.manual_seed(agent_init['seed'])
        self.discount_factor = agent_init["discount_factor"]
        self.num_action = agent_init["num_action"]
        self.mini_batch_size = agent_init["mini_batch_size"]
        self.experience = namedtuple('Experience', field_names=["state", "action", "action_prob", "reward", "done"])
        self.epsilon = agent_init["epsilon"]
        self.memory = list()
        self.actor = PPOActor(agent_init["actor_init"], optimizer_args=agent_init["optim_actor"])
        self.critic = PPOCritic(agent_init["critic_init"], optimizer_args=agent_init["optim_critic"])
        self.num_epoch = agent_init["num_epoch"]
        self.random_generator = np.random.RandomState(seed=agent_init['seed'])
        self.last_state = None
        self.last_action = None
        self.last_action_prob = None

    def append_experience(self, state, action, action_prob, reward, done):
        episode = self.experience(state, action, action_prob, reward, done)
        self.memory.append(episode)

    def episode_init(self):
        pass

    def policy(self, state):
        action_probs = self.actor.predict(state)
        action = self.random_generator.choice(self.num_action, p=action_probs)

        self.last_state = state
        self.last_action = action
        self.last_action_prob = action_probs[action]

        return action

    def compute_discounted_rewards(self, rewards, dones):
        discounted_reward = 0.0
        discounted_rewards = list()
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0.0
            discounted_reward = reward + self.discount_factor * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        return discounted_rewards

    def update(self, reward, done):
        self.append_experience(self.last_state, self.last_action, self.last_action_prob, reward, done)

        if len(self.memory) < self.mini_batch_size:
            return

        old_states, old_actions, old_action_probs, rewards, dones = list(zip(*self.memory))
        discounted_rewards = self.compute_discounted_rewards(rewards, dones)

        discounted_rewards = utils.to_tensor(discounted_rewards)
        old_action_probs = utils.to_tensor(old_action_probs)

        for _ in range(self.num_epoch):
            action_probs = self.actor.compute_probs(old_states, old_actions)
            state_values = self.critic.predict(old_states)
            r = action_probs/old_action_probs
            advantage_values = discounted_rewards - state_values
            actor_loss = -torch.min(
                r*advantage_values,
                torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantage_values
            )  # here add the entropy
            critic_loss = self.critic.loss(state_values.squeeze(), discounted_rewards)

            # LEARN PYTORCH
            loss = actor_loss.mean() + critic_loss
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.actor.nn_old.load_state_dict(self.actor.nn.state_dict())
        self.memory = list()


class PPOActor:
    """
    Actor using Proximal Policy Optimization with Adaptive KL penalty along with SGD to optimize its policy
    """
    def __init__(self, actor_init, optimizer_args={}):
        torch.manual_seed(actor_init['seed'])
        self.entropy_learning_rate = actor_init['entropy_learning_rate']
        self.nn = utils.init_nn(actor_init["nn_archi"])
        self.nn_old = utils.init_nn(actor_init["nn_archi"])
        self.optimizer = torch.optim.Adam(self.nn.parameters(), **optimizer_args)
        self.loss_history = list()

    def predict(self, state):  # return action's distribution
        state = utils.to_tensor(state)
        actions_probs = self.nn_old(state)
        return actions_probs.data.numpy()

    def compute_probs(self, states, actions):  # return probability for each action
        if not isinstance(states, torch.Tensor):
            states = utils.to_tensor(states)
        if not isinstance(actions, torch.Tensor):
            actions = utils.to_tensor(actions).long()
        # compute entropy here
        action_probs = self.nn(states)
        return action_probs.gather(1, actions.unsqueeze(1))


class PPOCritic:
    """
    Value based Critic (used for A2C)
    """
    def __init__(self, critic_init, optimizer_args={}):
        torch.manual_seed(critic_init['seed'])
        self.nn = utils.init_nn(critic_init["nn_archi"])
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), **optimizer_args)
        self.loss_history = list()

    def predict(self, state):  # return a value
        state = utils.to_tensor(state)
        return self.nn(state)
