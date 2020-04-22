import gym
import numpy as np
import tiles3 as tc


class TileCoder:
    def __init__(self, num_tiles, num_tilings, hash_size, position_boundaries, velocity_boundaries):
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.iht = tc.IHT(hash_size)

        self.position_scale = self.num_tiles / (position_boundaries[1] - position_boundaries[0])
        self.velocity_scale = self.num_tiles / (velocity_boundaries[1] - velocity_boundaries[0])

    def get_active_tiles(self, state):
        position, velocity = state
        state_scaled = [position * self.position_scale, velocity * self.velocity_scale]

        active_tiles = tc.tiles(self.iht, self.num_tilings, state_scaled)

        return np.array(active_tiles)


class Agent:

    def __init__(self, agent_init):

        self.discount_factor = agent_init["discount_factor"]
        self.learning_rate = agent_init["learning_rate"]/agent_init["tile_coder"]["num_tilings"]
        self.epsilon = agent_init["epsilon"]
        self.num_action = agent_init["num_action"]
        self.w = np.zeros((agent_init["num_action"], agent_init["tile_coder"]["hash_size"]))
        self.tile_coder = TileCoder(*agent_init["tile_coder"].values())

    def e_greedy(self, active_tiles):

        action_values = np.zeros(self.num_action)
        for action in range(self.num_action):
            action_values[action] = np.sum(self.w[action][active_tiles])

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_action)
        else:
            action = np.argmax(action_values)

        return action

    def choose_action(self, state):
        pass

    def update(self, state, reward, done):
        if not done:
            self.update_step(state, reward)
        if done:
            self.update_end(state, reward)

    def update_step(self, state, reward):
        pass

    def update_end(self, state, reward):
        pass


class SarsaAgent(Agent):

    def __init__(self, agent_init):
        super().__init__(agent_init)
        self.next_tiles = None
        self.next_action = None

    def choose_action(self, state):

        active_tiles = self.tile_coder.get_active_tiles(state)
        action = self.e_greedy(active_tiles)

        self.next_action = action
        self.next_tiles = active_tiles

        return action

    def update_step(self, next_state, reward):

        current_action = self.next_action
        current_tiles = self.next_tiles

        self.choose_action(next_state)

        target = reward + self.discount_factor * np.sum(self.w[self.next_action, self.next_tiles]) \
                 - np.sum(self.w[current_action, current_tiles])
        self.w[current_action, current_tiles] += self.learning_rate * target

    def update_end(self, state, reward):
        current_action = self.next_action
        current_tiles = self.next_tiles

        target = reward - np.sum(self.w[current_action][current_tiles])
        self.w[current_action][current_tiles] += self.learning_rate * target


class QLearningAgent(Agent):

    def __init__(self, agent_init):
        super().__init__(agent_init)
        self.current_tiles = None
        self.current_action = None

    def choose_action(self, state):

        if self.current_tiles is None:
            active_tiles = self.tile_coder.get_active_tiles(state)
        else:
            active_tiles = self.current_tiles

        action = self.e_greedy(active_tiles)

        if self.current_tiles is None:
            self.current_tiles = np.copy(active_tiles)

        self.current_action = action

        return action

    def update_step(self, next_state, reward):

        next_active_tiles = self.tile_coder.get_active_tiles(next_state)

        target = reward + self.discount_factor * np.max(self.w[:, next_active_tiles], axis=0).sum() \
                 - np.sum(self.w[self.current_action][self.current_tiles])
        self.w[self.current_action][self.current_tiles] += self.learning_rate * target

        self.current_tiles = next_active_tiles

    def update_end(self, state, reward):
        target = reward - np.sum(self.w[self.current_action][self.current_tiles])
        self.w[self.current_action][self.current_tiles] += self.learning_rate * target


class ExpectedSarsaAgent(Agent):

    def __init__(self, agent_init):
        super().__init__(agent_init)
        self.current_tiles = None
        self.current_action = None

    def choose_action(self, state):

        if self.current_tiles is None:
            active_tiles = self.tile_coder.get_active_tiles(state)
        else:
            active_tiles = self.current_tiles

        action = self.e_greedy(active_tiles)

        if self.current_tiles is None:
            self.current_tiles = np.copy(active_tiles)

        self.current_action = action

        return action

    def update_step(self, next_state, reward):

        next_active_tiles = self.tile_coder.get_active_tiles(next_state)

        target = reward + self.discount_factor * np.mean(self.w[:, next_active_tiles].sum(axis=1)) \
                 - np.sum(self.w[self.current_action][self.current_tiles])
        self.w[self.current_action][self.current_tiles] += self.learning_rate * target

        self.current_tiles = next_active_tiles

    def update_end(self, state, reward):
        target = reward - np.sum(self.w[self.current_action][self.current_tiles])
        self.w[self.current_action][self.current_tiles] += self.learning_rate * target

