import numpy as np
import utils


class Agent:
    def __init__(self, agent_init):
        self.discount_factor = agent_init["discount_factor"]
        self.learning_rate = agent_init["learning_rate"]
        self.exploration_handler = utils.ExplorationRateDecay(*agent_init["exploration_rate_decay"].values())
        self.num_action = agent_init["num_action"]
        self.num_state = agent_init["num_state"]
        self.q = np.ones((agent_init["num_action"], agent_init["num_state"]))
        self.random_generator = np.random.RandomState(seed=agent_init["seed"])
        self.state_dim = agent_init["state_dim"]
        self.next_action = None
        self.next_state = None

    def state_to_index(self, state):
        if type(state) is tuple:
            return np.ravel_multi_index(state, self.state_dim)
        return state

    def e_greedy(self, state, always_greedy=False):

        if self.random_generator.rand() < self.exploration_handler() and always_greedy is False:
            action = self.random_generator.randint(self.num_action)
        else:
            action = utils.argmax(self.q[:, state], random_generator=self.random_generator)

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
            self.exploration_handler.next()
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

        current_q = self.q[:, next_state]
        current_q_max = np.max(current_q)
        pi = np.ones(self.num_action) * (self.exploration_handler() / self.num_action)
        pi += (current_q == current_q_max) * ((1 - self.exploration_handler()) / np.sum(current_q == current_q_max))
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
        self.eligibility_traces = utils.EligibilityTraces(agent_init["trace_decay"],
                                                          (agent_init["num_action"], agent_init["num_state"]),
                                                          agent_init["eligibility_method"])

    def update_step(self, next_state, reward):
        current_action = self.next_action
        current_state = self.next_state

        next_action = self.e_greedy(next_state)

        target = reward + self.discount_factor * self.q[next_action, next_state] - self.q[current_action, current_state]

        self.eligibility_traces.update_traces(current_action, current_state)
        self.q += self.learning_rate * target * self.eligibility_traces()
        self.eligibility_traces.decay_traces(self.discount_factor)

        self.next_action = next_action
        self.next_state = next_state

        return next_action

    def update_end(self, reward):
        current_action = self.next_action
        current_state = self.next_state

        target = reward - self.q[current_action][current_state]

        self.eligibility_traces.update_traces(current_action, current_state)
        self.q += self.learning_rate * target * self.eligibility_traces()
        self.eligibility_traces.decay_traces(self.discount_factor)
