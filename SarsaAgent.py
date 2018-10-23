import numpy as np
from collections import defaultdict
from EpsilonDecay import EpsilonDecay

class SarsaAgent:
    def __init__(self, nA, alpha, epsilon):
        self.nA = nA
        self.alpha = alpha
        self.epsilon = epsilon
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
        # choose next action
        self.epsilon.decay()
        next_action = self.epsilon.greedy(self.Q[next_state])
        # update Q for last state and last action
        target = reward + self.Q[next_state][next_action]
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
        # update last state and last action
        self.last_state = next_state
        self.last_action = next_action
        return next_action