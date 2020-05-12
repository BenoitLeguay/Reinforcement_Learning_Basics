import numpy as np
import utils
import torch
import variable as v
from torch.distributions import Categorical
from copy import deepcopy


class A2CAgent:
    """
    Advantage Actor-Critic Agent
    """
    def __init__(self, agent_init):
        self.discount_factor = agent_init["discount_factor"]
        self.num_action = agent_init["num_action"]
        self.actor = SGDActor(agent_init["actor_init"], optimizer_args=agent_init["optim_actor"])
        self.critic = VCritic(agent_init["critic_init"], optimizer_args=agent_init["optim_critic"])
        self.random_generator = np.random.RandomState(seed=agent_init['seed'])
        self.next_state = None
        self.next_action = None

    @staticmethod
    def flatten_state(state):
        return np.ravel(state)

    def policy(self, state):
        action_probs = self.actor.predict(state)
        action = self.random_generator.choice(self.num_action, p=action_probs)
        return action

    def episode_init(self, state):
        state = self.flatten_state(state)

        action = self.policy(state)
        self.next_action = action
        self.next_state = state

        return action

    def update(self, state, reward, done):
        state = self.flatten_state(state)

        next_action = -1
        if not done:
            next_action = self.update_step(state, reward)
        if done:
            self.update_end(reward)

        return next_action

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state

        next_state_value = self.critic.predict(next_state)
        current_state_value = self.critic.predict(current_state)

        td_target = reward + self.discount_factor * next_state_value.data.numpy()
        td_error = td_target - current_state_value.data.numpy()

        self.critic.update(current_state_value, td_target)
        self.actor.update(current_state, current_action, td_error)

        next_action = self.policy(next_state)

        self.next_state = next_state
        self.next_action = next_action

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        current_state_value = self.critic.predict(current_state)
        td_target = np.array([float(reward)])
        td_error = td_target - current_state_value.data.numpy()
        self.critic.update(current_state_value, td_target)
        self.actor.update(current_state, current_action, td_error)


class SGDActor:
    """
    Actor using stochastic gradient descent to improve its policy
    """
    def __init__(self, actor_init, optimizer_args={}):
        torch.manual_seed(actor_init['seed'])
        self.entropy_learning_rate = actor_init['entropy_learning_rate']
        self.nn = utils.init_nn(actor_init["nn_archi"])
        self.optimizer = torch.optim.Adam(self.nn.parameters(), **optimizer_args)
        self.loss_history = list()

    def predict(self, state):  # return an action

        state = utils.to_tensor(state)
        actions_probs = self.nn(state)
        return actions_probs.data.numpy()

    def update(self, state, action, td_error):
        state = utils.to_tensor(state)
        action = utils.to_tensor(action).long()
        td_error = utils.to_tensor(td_error)

        actions_probs = self.nn(state)
        action_chosen_prob = torch.gather(actions_probs, dim=0, index=action)

        entropy = Categorical(probs=actions_probs).entropy()

        loss = -torch.log(action_chosen_prob) * td_error - self.entropy_learning_rate * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class VCritic:
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

    def update(self, current_state_value, td_target):
        td_target = utils.to_tensor(td_target)
        loss = self.loss(current_state_value, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class QACAgent(A2CAgent):
    """
    Q Actor-Critic Agent class
    """
    def __init__(self, agent_init):
        super().__init__(agent_init)
        self.critic = QCritic(agent_init["critic_init"])

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state
        next_action = self.policy(next_state)

        next_state_action_value = self.critic.predict(next_state, next_action)
        current_state_action_value = self.critic.predict(current_state, current_action)

        td_target = reward + self.discount_factor * next_state_action_value.data.numpy()
        td_error = td_target - current_state_action_value.data.numpy()

        self.critic.update(current_state_action_value, td_target)
        self.actor.update(current_state, current_action, td_error)

        self.next_state = next_state
        self.next_action = next_action

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        current_state_action_value = self.critic.predict(current_state, current_action)
        td_target = np.array(float(reward))
        td_error = td_target - current_state_action_value.data.numpy()
        self.critic.update(current_state_action_value, td_target)
        self.actor.update(current_state, current_action, td_error)


class QCritic:
    """
    Action-Value based Critic (used for QAC)
    """
    def __init__(self, critic_init, optimizer_args={}):
        torch.manual_seed(critic_init['seed'])
        self.nn = utils.init_nn(critic_init["nn_archi"])
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), **optimizer_args)
        self.loss_history = list()

    def predict(self, state, action):  # return a value
        state = utils.to_tensor(state)
        action = utils.to_tensor(action).long()
        action_values = self.nn(state)
        action_value = torch.gather(action_values, dim=0, index=action)
        return action_value

    def update(self, current_state_value, td_target):
        td_target = utils.to_tensor(td_target)
        loss = self.loss(current_state_value, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class PPOAgent:
    

class PPOAKLActor:
    """
    Actor using Proximal Policy Optimization with Adaptive KL penalty along with SGD to optimize its policy
    """
    pass


class PPOCOActor:
    """
    Actor using Proximal Policy Optimization with Clipped Objective along with SGD to optimize its policy
    """
    pass
