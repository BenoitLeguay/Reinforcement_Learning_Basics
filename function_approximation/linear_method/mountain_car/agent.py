import numpy as np
import utils
import variable as var

#  https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5092/5494
#  http://incompleteideas.net/book/first/ebook/node89.html


class Agent:

    def __init__(self, agent_init):

        self.discount_factor = agent_init["discount_factor"]
        self.learning_rate = agent_init["learning_rate"]/agent_init["tile_coder"]["num_tilings"]
        self.exploration_handler = utils.ExplorationRateDecay(*agent_init["exploration_rate"].values())
        self.num_action = agent_init["num_action"]
        self.w = np.zeros((agent_init["num_action"], agent_init["tile_coder"]["hash_size"]))
        self.tile_coder = utils.TileCoder(*agent_init["tile_coder"].values())
        self.max_position = agent_init["max_position_init"]
        self.max_position_reward_bonus = agent_init["max_position_reward_bonus"]
        self.random_generator = np.random.RandomState(seed=agent_init["seed"])
        self.next_tiles = None
        self.next_action = None

    def e_greedy(self, active_tiles):

        action_values = np.zeros(self.num_action)
        for action in range(self.num_action):
            action_values[action] = np.sum(self.w[action][active_tiles])

        if self.random_generator.rand() < self.exploration_handler():
            action = self.random_generator.randint(self.num_action)
        else:
            action = utils.argmax(action_values, random_generator=self.random_generator)

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
            self.exploration_handler.next()
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

        target = reward + self.discount_factor * np.max(self.w[:, next_active_tiles].sum(axis=1)) \
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

        current_q = self.w[:, next_active_tiles].sum(axis=1)
        q_max = np.max(current_q)
        pi = np.ones(self.num_action) * (self.exploration_handler() / self.num_action)
        pi += (current_q == q_max) * ((1 - self.exploration_handler()) / np.sum(current_q == q_max))
        expectation = np.sum(current_q * pi)

        target = reward + self.discount_factor * expectation - np.sum(self.w[current_action][current_tiles])
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
        self.eligibility_traces = utils.EligibilityTraces(agent_init["trace_decay"],
                                                          (agent_init["num_action"], agent_init["tile_coder"]["hash_size"]),
                                                          agent_init["eligibility_method"])

    def episode_init(self, state):

        self.eligibility_traces.episode_reset()

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

        td_error = reward + self.discount_factor * np.sum(self.w[next_action, next_tiles]) - np.sum(self.w[current_action, current_tiles])

        self.eligibility_traces.update_traces(current_action, current_tiles)

        self.w[current_action, current_tiles] += self.learning_rate * td_error * self.eligibility_traces()[current_action, current_tiles]

        self.eligibility_traces.decay_traces(self.discount_factor, current_action, current_tiles)

        self.next_action = next_action
        self.next_tiles = next_tiles

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_tiles = self.next_tiles

        td_error = reward - np.sum(self.w[current_action, current_tiles])

        self.eligibility_traces.update_traces(current_action, current_tiles)

        self.w[current_action, current_tiles] += self.learning_rate * td_error * self.eligibility_traces()[current_action, current_tiles]

