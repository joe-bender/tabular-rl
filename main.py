import gym
from collections import deque
import numpy as np
from EpsilonDecay import EpsilonDecay
from QLearningAgent import QLearningAgent

environment = 'Taxi-v2'
n_episodes = 1_000_000
avg_window = 100
solution = 9.7

alpha = 0.4
epsilon = EpsilonDecay(1, 0, .99)

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
    latest_scores.append(score)
    if i_episode > avg_window:
        avg_score = np.mean(latest_scores)
        print('Episode: {}, '.format(i_episode), end='')
        print('Avg Score: {:7.2f}, '.format(avg_score), end='')
        print('Epsilon: {:7.5f}'.format(epsilon.value))
        if avg_score >= solution:
            print('Solved in {} episodes'.format(i_episode))
            break
