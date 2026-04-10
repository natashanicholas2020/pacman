import sys
import time
import pickle
import numpy as np
# from tqdm import tqdm
from vis_pacman import *
import matplotlib.pyplot as plt

game = env

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    Q_table = {}
    N = {}
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, reward, done, info = game.reset()
        state = hash(obs)
        total_reward = 0

        while not done:
            if state not in Q_table:

                #initialze in table for all actions
                Q_table[state] = np.zeros(game.action_space.n)
                N[state] = np.zeros(game.action_space.n)

            if np.random.rand() < epsilon: # explore: pick a random action
                action = np.random.randint(game.action_space.n)

            else: # exploit: pick the best Q-value action
                action = np.argmax(Q_table[state])

            #determines what happens at next action
            next_obs, reward, done, info = game.step(action)
            next_state = hash(next_obs)
            total_reward += reward

            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(game.action_space.n)
                N[next_state] = np.zeros(game.action_space.n)

            alpha = 1/(1 + N[state][action]) #if using a learning schedule

            max_next_q = np.max(Q_table[next_state])  # best possible value in next state, used to be just max

            #q-learning equation
            Q_table[state][action] += alpha * (reward + gamma * max_next_q - Q_table[state][action])

            #update n
            N[state][action] += 1

            state = next_state
            obs = next_obs
        episode_rewards.append(total_reward)
        epsilon *= decay_rate

    return Q_table, N, episode_rewards
    #pass