import random
from collections import defaultdict
import numpy as np

def Q_learning(env, num_episodes=5000, gamma=0.9, epsilon=1.0, decay_rate=0.999, alpha=0.1):

    Q = defaultdict(float)
    actions = [0, 1, 2, 3]  # fixed action space

    for episode in range(num_episodes):
        obs, _, _, _ = env.reset()
        state = simplify_state(obs) 
        done = False
        episode_reward = 0

        while not done:

            # ε-greedy
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(actions, key=lambda a: Q[(state, a)])

            next_obs, reward, done, info = env.step(action)
            next_state = simplify_state(next_obs)

            episode_reward += reward

            # Q-learning update
            if done:
                max_next_Q = 0
            else:
                max_next_Q = max(Q[(next_state, a)] for a in actions)

            Q[(state, action)] += alpha * (
                reward + gamma * max_next_Q - Q[(state, action)]
            )

            state = next_state

        if(episode % 100 == 0):
            print(episode, ":", episode_reward)

        epsilon *= decay_rate

    return Q

def simplify_state(state):
    px, py = state["pacman_position"]

    # Convert pellet grid back to coordinates
    pellets = state["pellet_positions"]
    grid_w = int(np.sqrt(len(pellets)))  # assumes square grid

    pellet_coords = [
        (i // grid_w, i % grid_w)
        for i, v in enumerate(pellets) if v == 1
    ]

    # Ghost positions (dict → list of tuples)
    ghosts = list(state["ghost_positions"].values())

    # distance to nearest pellet
    if pellet_coords:
        nearest_pellet = min(
            pellet_coords,
            key=lambda p: abs(px - p[0]) + abs(py - p[1])
        )
        food_dx = nearest_pellet[0] - px
        food_dy = nearest_pellet[1] - py
    else:
        food_dx, food_dy = 0, 0

    # distance to nearest ghost
    if ghosts:
        nearest_ghost = min(
            ghosts,
            key=lambda g: abs(px - g[0]) + abs(py - g[1])
        )
        ghost_dx = nearest_ghost[0] - px
        ghost_dy = nearest_ghost[1] - py
    else:
        ghost_dx, ghost_dy = 0, 0

    return (
        px, py,
        food_dx, food_dy,
        ghost_dx, ghost_dy
    )

# import sys
# import time
# import pickle
# import numpy as np
# # from tqdm import tqdm
# from vis_pacman import *
# import matplotlib.pyplot as plt

# game = env

# def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
#     Q_table = {}
#     N = {}
#     episode_rewards = []
    
#     for ep in range(num_episodes):
#         obs, reward, done, info = game.reset()
#         state = hash(obs)
#         total_reward = 0

#         while not done:
#             if state not in Q_table:

#                 #initialze in table for all actions
#                 Q_table[state] = np.zeros(game.action_space.n)
#                 N[state] = np.zeros(game.action_space.n)

#             if np.random.rand() < epsilon: # explore: pick a random action
#                 action = np.random.randint(game.action_space.n)

#             else: # exploit: pick the best Q-value action
#                 action = np.argmax(Q_table[state])

#             #determines what happens at next action
#             next_obs, reward, done, info = game.step(action)
#             next_state = hash(next_obs)
#             total_reward += reward

#             if next_state not in Q_table:
#                 Q_table[next_state] = np.zeros(game.action_space.n)
#                 N[next_state] = np.zeros(game.action_space.n)

#             alpha = 1/(1 + N[state][action]) #if using a learning schedule

#             max_next_q = np.max(Q_table[next_state])  # best possible value in next state, used to be just max

#             #q-learning equation
#             Q_table[state][action] += alpha * (reward + gamma * max_next_q - Q_table[state][action])

#             #update n
#             N[state][action] += 1

#             state = next_state
#             obs = next_obs
#         episode_rewards.append(total_reward)
#         epsilon *= decay_rate

#     return Q_table, N, episode_rewards
#     #pass