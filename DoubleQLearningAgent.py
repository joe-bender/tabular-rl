import numpy as np
from collections import defaultdict
from EpsilonDecay import EpsilonDecay

class DoubleQLearningAgent:
    def __init__(self, nA, alpha, epsilon, gamma):
        self.nA = nA
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q1 = defaultdict(lambda: np.zeros(self.nA))
        self.Q2 = defaultdict(lambda: np.zeros(self.nA))

    def reset(self, state):
        self.last_state = state
        action = self.epsilon.greedy(self.Q1[state])
        self.last_action = action
        return action

    def step(self, next_state, reward, done):
        # update previous timestep
        state = self.last_state
        action = self.last_action
        # flip coin to choose which Q to update
        if np.random.rand() > .5:
            Q1 = self.Q1
            Q2 = self.Q2
        else:
            Q1 = self.Q2
            Q2 = self.Q1
        a = np.argmax(Q1[next_state])
        target = reward + self.gamma * Q2[next_state][a]
        Q1[state][action] += self.alpha * (target - Q1[state][action])
        # choose next action using average of Q values
        self.epsilon.decay()
        stack = np.stack([self.Q1[next_state], self.Q2[next_state]])
        row_mean = np.mean(stack, axis=0)
        assert(len(row_mean) == self.nA)
        action = self.epsilon.greedy(row_mean)
        # update last state and last action
        self.last_state = next_state
        self.last_action = action
        return action
