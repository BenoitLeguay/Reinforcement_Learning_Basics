import numpy as np
import torch
import variable as v
import utils


class Agent:
    def __init__(self, agent_init):
        self.exploration_handler = utils.ExplorationRateDecay(*agent_init["exploration_rate"].values())
        self.num_action = agent_init["num_action"]
        self.nn_handler = utils.NeuralNetworkHandler(agent_init["neural_network_handler"], agent_init['optim'], agent_init['replay_buffer'])
        self.max_position = agent_init["max_position_init"]
        self.max_position_reward_bonus = agent_init["max_position_reward_bonus"]
        self.random_generator = np.random.RandomState(seed=agent_init['seed'])
        self.next_state = None
        self.next_action = None
        self.next_values = None

    def e_greedy(self, state):

        state_tensor = self.nn_handler.to_float_tensor(state)
        values = self.nn_handler.eval_nn(state_tensor).detach()

        if self.random_generator.rand() < self.exploration_handler():
            action = self.random_generator.randint(self.num_action)
        else:

            action = values.max(0)[1].item()

        return action

    def max_position_reward_function(self, new_position, reward):
        if new_position > self.max_position:
            self.max_position = new_position
            reward += self.max_position_reward_bonus

        return reward

    def episode_init(self, state):
        action = self.e_greedy(state)
        self.next_action = action
        self.next_state = state

        return action

    def update(self, state, reward, done):

        reward = self.max_position_reward_function(state[0], reward)

        next_action = -1
        if not done:
            next_action = self.update_step(state, reward)
        if done:
            self.exploration_handler.next()
            self.update_end(reward)

        return next_action

    def update_step(self, state, reward):
        pass

    def update_end(self, reward):
        pass


class DQN(Agent):

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state

        self.nn_handler.update_step(current_state, current_action, reward, next_state, 0)

        next_action = self.e_greedy(next_state)

        self.next_state = next_state
        self.next_action = next_action

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        self.nn_handler.update_step(current_state, current_action, reward, current_state, 1)
