import numpy as np
from collections import defaultdict
from EpsilonDecay import EpsilonDecay

class MCAgent:
    def __init__(self, nA, alpha, epsilon):
        self.nA = nA
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.wins = 0

    def reset(self, state):
        self.states = []
        self.actions = []
        self.rewards = []
        self.states.append(state)
        action = self.epsilon.greedy(self.Q[state])
        self.actions.append(action)
        return action
    
    def step(self, next_state, reward, done):
        self.rewards.append(reward)
        if done:
            self.updateQ()
            return None
        else:
            self.states.append(next_state)
            self.epsilon.decay()
            action = self.epsilon.greedy(self.Q[next_state])
            self.actions.append(action)
            return action
    
    def updateQ(self):
        assert(len(self.states) == len(self.actions) == len(self.rewards))
        episode_steps = len(self.states)
        visited = set()
        for t in range(episode_steps):
            state, action = self.states[t], self.actions[t]
            if (state, action) not in visited:
                G = sum(self.rewards[t:])
                self.Q[state][action] += self.alpha * (G - self.Q[state][action])
                visited.add((state, action))