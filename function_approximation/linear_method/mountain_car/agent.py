import numpy as np
import utils
import variable as var


np.random.seed(42)
#  https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5092/5494
#  http://incompleteideas.net/book/first/ebook/node89.html


class Agent:

    def __init__(self, agent_init):

        self.discount_factor = agent_init["discount_factor"]
        self.learning_rate = agent_init["learning_rate"]/agent_init["tile_coder"]["num_tilings"]
        self.epsilon = utils.ExplorationRateDecay(*agent_init["exploration_rate"].values())
        self.num_action = agent_init["num_action"]
        self.w = np.zeros((agent_init["num_action"], agent_init["tile_coder"]["hash_size"]))
        self.tile_coder = utils.TileCoder(*agent_init["tile_coder"].values())
        self.max_position = agent_init["max_position_init"]
        self.max_position_reward_bonus = agent_init["max_position_reward_bonus"]
        self.next_tiles = None
        self.next_action = None

    def e_greedy(self, active_tiles):

        action_values = np.zeros(self.num_action)
        for action in range(self.num_action):
            action_values[action] = np.sum(self.w[action][active_tiles])

        if np.random.rand() < self.epsilon.er:
            action = np.random.randint(self.num_action)
        else:
            action = utils.argmax(action_values)

        return action

    def max_position_reward_function(self, new_position, reward):
        if new_position > self.max_position:
            self.max_position = new_position
            reward += self.max_position_reward_bonus

        return reward

    def episode_init(self, state):

        active_tiles = self.tile_coder.get_active_tiles(state)
        action = self.e_greedy(active_tiles)

        self.next_action = action
        self.next_tiles = active_tiles

        return action

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


class SarsaAgent(Agent):

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_tiles = self.next_tiles

        next_tiles = self.tile_coder.get_active_tiles(next_state)
        next_action = self.e_greedy(next_tiles)

        target = reward + self.discount_factor * np.sum(self.w[next_action, next_tiles]) \
                 - np.sum(self.w[current_action, current_tiles])
        self.w[current_action, current_tiles] += self.learning_rate * target

        self.next_action = next_action
        self.next_tiles = next_tiles

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_tiles = self.next_tiles

        target = reward - np.sum(self.w[current_action][current_tiles])
        self.w[current_action][current_tiles] += self.learning_rate * target


class QLearningAgent(Agent):

    def update_step(self, next_state, reward):

        current_action = self.next_action
        current_tiles = self.next_tiles

        next_active_tiles = self.tile_coder.get_active_tiles(next_state)

        target = reward + self.discount_factor * np.max(self.w[:, next_active_tiles], axis=0).sum() \
                 - np.sum(self.w[current_action][current_tiles])
        self.w[current_action][current_tiles] += self.learning_rate * target

        self.next_tiles = np.copy(next_active_tiles)
        self.next_action = self.e_greedy(next_active_tiles)

        return self.next_action

    def update_end(self, reward):

        current_action = self.next_action
        current_tiles = self.next_tiles

        target = reward - np.sum(self.w[current_action][current_tiles])
        self.w[current_action][current_tiles] += self.learning_rate * target


class ExpectedSarsaAgent(Agent):

    def update_step(self, next_state, reward):

        current_action = self.next_action
        current_tiles = self.next_tiles

        next_active_tiles = self.tile_coder.get_active_tiles(next_state)

        target = reward + self.discount_factor * np.mean(self.w[:, next_active_tiles].sum(axis=1)) \
                 - np.sum(self.w[current_action][current_tiles])
        self.w[current_action][current_tiles] += self.learning_rate * target

        self.next_tiles = np.copy(next_active_tiles)
        self.next_action = self.e_greedy(next_active_tiles)

        return self.next_action

    def update_end(self, reward):

        current_action = self.next_action
        current_tiles = self.next_tiles

        target = reward - np.sum(self.w[current_action][current_tiles])
        self.w[current_action][current_tiles] += self.learning_rate * target


class SarsaLambdaAgent(Agent):
    def __init__(self, agent_init):
        super().__init__(agent_init)
        self.learning_rate = utils.AdaptiveLearningRate()
        self.trace_decay = agent_init["trace_decay"]
        self.eligibility = np.zeros((agent_init["num_action"], agent_init["tile_coder"]["hash_size"]))
        self.eligibility_method = agent_init["eligibility_method"]

    def episode_init(self, state):

        self.eligibility = np.zeros(self.eligibility.shape)

        active_tiles = self.tile_coder.get_active_tiles(state)
        action = self.e_greedy(active_tiles)

        self.next_action = action
        self.next_tiles = active_tiles

        return action

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_tiles = self.next_tiles

        next_tiles = self.tile_coder.get_active_tiles(next_state)
        next_action = self.e_greedy(next_tiles)

        td_error = reward
        td_error -= np.sum(self.w[current_action, current_tiles])

        self.update_eligibility(current_action, current_tiles)

        td_error += self.discount_factor * np.sum(self.w[next_action, next_tiles])
        eligibility_mask = np.zeros(self.tile_coder.hash_size, dtype=float)
        eligibility_mask[current_tiles] = 1.0

        self.learning_rate.next(self.tile_coder.index_feature_to_mask_feature(current_tiles),
                                self.tile_coder.index_feature_to_mask_feature(next_tiles),
                                eligibility_mask,
                                self.discount_factor)

        self.w[current_action, current_tiles] += self.learning_rate.epsilon * td_error * self.eligibility[current_action, current_tiles]
        self.eligibility[current_action, current_tiles] *= self.discount_factor * self.trace_decay

        self.next_action = next_action
        self.next_tiles = next_tiles

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_tiles = self.next_tiles

        td_error = reward

        td_error -= np.sum(self.w[current_action, current_tiles])

        self.update_eligibility(current_action, current_tiles)

        self.w[current_action, current_tiles] += self.learning_rate.l * td_error * self.eligibility[current_action, current_tiles]

    def update_eligibility(self, current_action, current_tiles):

        if self.eligibility_method == 'accumulate':
            self.eligibility[current_action, current_tiles] += 1
        elif self.eligibility_method == 'replace':
            self.eligibility[current_action, current_tiles] = 1
