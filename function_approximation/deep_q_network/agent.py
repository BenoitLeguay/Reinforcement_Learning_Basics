import numpy as np
import torch
import variable as v
import utils


class Agent:
    def __init__(self, agent_init):
        self.discount_factor = agent_init["discount_factor"]
        self.exploration_handler = utils.ExplorationRateDecay(*agent_init["exploration_rate"].values())
        self.num_action = agent_init["num_action"]
        self.nn_handler = utils.NeuralNetworkHandler(agent_init["nn_archi"], agent_init["learning_rate"])
        self.max_position = agent_init["max_position_init"]
        self.max_position_reward_bonus = agent_init["max_position_reward_bonus"]
        self.next_state = None
        self.next_action = None
        self.next_values = None

    @staticmethod
    def init_nn(nn_archi):

        layers = list()

        for layer_args in nn_archi:
            layer_type = layer_args['type']
            in_features, out_features, activation = layer_args["in"], layer_args["out"], layer_args["activation"]
            layers.append(v.layer_name_to_obj[layer_type](in_features, out_features))
            layers.append(v.activation_name_to_obj[activation]())

        dqn = torch.nn.Sequential(*layers)
        return dqn

    def e_greedy(self, state_tensor):

        values = self.nn_handler.nn(state_tensor).detach()

        if np.random.rand() < self.exploration_handler.rate:
            action = np.random.randint(self.num_action)
        else:

            action = values.max(0)[1].item()

        return action, values

    def max_position_reward_function(self, new_position, reward):
        if new_position > self.max_position:
            self.max_position = new_position
            reward += self.max_position_reward_bonus

        return reward

    def episode_init(self, state):
        state_tensor = self.nn_handler.state_to_tensor(state)
        action_tensor, next_values = self.e_greedy(state_tensor)
        self.next_action = action_tensor
        self.next_state = state_tensor
        self.next_values = next_values

        return action_tensor

    def update(self, state, reward, done):

        next_action = -1
        if not done:
            next_action = self.update_step(state, reward)
        if done:
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
        current_values = self.next_values

        next_state_tensor = self.nn_handler.state_to_tensor(next_state)
        next_action, next_values = self.e_greedy(next_state_tensor)

        expected_values = current_values.clone()

        # if sarsa, take next_values[next_action]; if q learning, take next_values.max(0)[1]
        expected_values[current_action] = reward + self.discount_factor * next_values.max(0)[0]

        self.nn_handler.update_nn(current_values, expected_values)

        self.next_action = next_action
        self.next_state = next_state_tensor
        self.next_values = next_values

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state
        current_values = self.next_values

        expected_values = current_values.clone()
        expected_values[current_action] = reward

        self.nn_handler.update_nn(current_values, expected_values)
