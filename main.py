import gym
from collections import deque
import numpy as np
from EpsilonDecay import EpsilonDecay
from QLearningAgent import QLearningAgent

environment = 'Taxi-v2'
n_episodes = 1_000_000
avg_window = 100

alpha = 0.2
epsilon = EpsilonDecay(1, 0, .9999)

#-------------------------------------------------------------------------------

env = gym.make(environment)
nA = env.action_space.n
agent = QLearningAgent(nA, alpha, epsilon)

latest_scores = deque(maxlen=avg_window)
for i_episode in range(1, n_episodes+1):
    state = env.reset()
    action = agent.reset(state)
    score = 0
    # episode loop
    while True:
        next_state, reward, done, _ = env.step(action)
        score += reward
        action = agent.step(next_state, reward, done)
        if done:
            break
    # print info
    avg_score = np.mean(latest_scores)
    if i_episode > avg_window:
        print('Episode: {}, '.format(i_episode), end='')
        print('Avg Score: {}, '.format(score), end='')
        print('Epsilon: {}'.format(epsilon.value))
