import numpy as np


class Agent:
    def __init__(self, agent_init):
        self.discount_factor = agent_init["discount_factor"]
        self.learning_rate = agent_init["learning_rate"]
        self.epsilon = ExplorationRateDecay(*agent_init["exploration_rate_decay"].values())
        self.num_action = agent_init["num_action"]
        self.num_state = agent_init["num_state"]
        self.q = np.ones((agent_init["num_action"], agent_init["num_state"]))
        self.next_action = None
        self.next_state = None

    def e_greedy(self, state):

        if np.random.rand() < self.epsilon.er:
            action = np.random.randint(self.num_action)
        else:
            action = np.argmax(self.q[:, state])    # TODO: change to tie break argmax function

        return action

    def episode_init(self, state):

        action = self.e_greedy(state)

        self.next_action = action
        self.next_state = state

        return action

    def update(self, state, reward, done):
        next_action = -1
        if not done:
            next_action = self.update_step(state, reward)
        if done:
            self.update_end(reward)

        return next_action

    def update_step(self, next_state, reward):
        pass

    def update_end(self, reward):
        pass


class SarsaAgent(Agent):

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state

        next_action = self.e_greedy(next_state)

        target = reward + self.discount_factor * self.q[next_action, next_state] - self.q[current_action, current_state]
        self.q[current_action, current_state] += self.learning_rate * target

        self.next_action = next_action
        self.next_state = next_state

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        target = reward - self.q[current_action][current_state]
        self.q[current_action][current_state] += self.learning_rate * target


class QLearningAgent(Agent):

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state

        target = reward + self.discount_factor * np.max(self.q[:, next_state]) - self.q[current_action, current_state]
        self.q[current_action, current_state] += self.learning_rate * target

        next_action = self.e_greedy(next_state)

        self.next_action = next_action
        self.next_state = next_state

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        target = reward - self.q[current_action][current_state]
        self.q[current_action][current_state] += self.learning_rate * target


class ExpectedSarsaAgent(Agent):

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state

        current_q = self.q[:, current_state]
        current_q_max = np.max(current_q)
        pi = np.ones(self.num_action) * (self.epsilon.er / self.num_action)
        pi += (current_q == current_q_max) * ((1 - self.epsilon.er) / np.sum(current_q == current_q_max))
        expectation = np.sum(current_q * pi)

        target = reward + self.discount_factor * expectation - self.q[current_action, current_state]
        self.q[current_action, current_state] += self.learning_rate * target

        next_action = self.e_greedy(next_state)

        self.next_action = next_action
        self.next_state = next_state

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        target = reward - self.q[current_action][current_state]
        self.q[current_action][current_state] += self.learning_rate * target


class SarsaLambdaAgent(Agent):
    def __init__(self, agent_init):
        super(SarsaLambdaAgent, self).__init__(agent_init)
        self.eligibility = np.zeros((agent_init["num_action"], agent_init["num_state"]))
        self.trace_decay = agent_init["trace_decay"]

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state

        next_action = self.e_greedy(next_state)

        target = reward + self.discount_factor * self.q[next_action, next_state] - self.q[current_action, current_state]
        self.eligibility[current_action, current_state] += 1

        for action in range(self.num_action):
            for state in range(self.num_state):
                self.q[action, state] += self.learning_rate * target * self.eligibility[action, state]
                self.eligibility[action, state] *= self.discount_factor * self.trace_decay

        self.next_action = next_action
        self.next_state = next_state

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        target = reward - self.q[current_action][current_state]
        self.eligibility[current_action, current_state] += 1

        for action in range(self.num_action):
            for state in range(self.num_state):
                self.q[action, state] += self.learning_rate * target * self.eligibility[action, state]
                self.eligibility[action, state] *= self.discount_factor * self.trace_decay


class ExplorationRateDecay:
    def __init__(self, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, constant_exploration_rate):
        self.er = exploration_rate
        self.max_er = max_exploration_rate
        self.min_er = min_exploration_rate
        self.decay_er = exploration_decay_rate
        self.episode_count = 0.0
        self.constant_er = constant_exploration_rate

    def next(self):

        if self.constant_er:
            return self.er

        self.er = self.min_er + ((self.max_er - self.min_er) * (np.exp(-self.decay_er * self.episode_count)))
        self.episode_count += 1.0

        return self.er

    def reset_episode_count(self):
        self.episode_count = 0.0

