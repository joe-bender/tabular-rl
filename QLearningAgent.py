import numpy as np
from collections import defaultdict
from EpsilonDecay import EpsilonDecay

class QLearningAgent:
    def __init__(self, nA, alpha, epsilon, gamma):
        self.nA = nA
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def reset(self, state):
        self.last_state = state
        action = self.epsilon.greedy(self.Q[state])
        self.last_action = action
        return action

    def step(self, next_state, reward, done):
        # update previous timestep
        state = self.last_state
        action = self.last_action
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
        # choose next action
        self.epsilon.decay()
        action = self.epsilon.greedy(self.Q[next_state])
        # update last state and last action
        self.last_state = next_state
        self.last_action = action
        return action
