import random
from collections import defaultdict
import numpy as np

def Q_learning(env, num_episodes=5000, gamma=0.95, epsilon=1.0, decay_rate=0.9998, alpha=0.2):

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

        epsilon = max(0.05, epsilon * decay_rate)

    return Q
def simplify_state(state):
    px, py = state["pacman_position"]
    pellets = state["pellet_positions"]
    grid_w = int(np.sqrt(len(pellets)))

    pellet_coords = [(i // grid_w, i % grid_w) for i, v in enumerate(pellets) if v == 1]
    ghosts = list(state["ghost_positions"].values())

    if pellet_coords:
        nearest = min(pellet_coords, key=lambda p: abs(px-p[0]) + abs(py-p[1]))
        food_dx = np.sign(nearest[0] - px)   # -1, 0, or 1
        food_dy = np.sign(nearest[1] - py)
        food_dist = min(abs(px-nearest[0]) + abs(py-nearest[1]), 5)  # capped at 5
    else:
        food_dx, food_dy, food_dist = 0, 0, 0

    if ghosts:
        nearest_g = min(ghosts, key=lambda g: abs(px-g[0]) + abs(py-g[1]))
        ghost_dx = np.sign(nearest_g[0] - px)
        ghost_dy = np.sign(nearest_g[1] - py)
        ghost_dist = min(abs(px-nearest_g[0]) + abs(py-nearest_g[1]), 8)  # capped at 8
    else:
        ghost_dx, ghost_dy, ghost_dist = 0, 0, 8

    return (food_dx, food_dy, food_dist, ghost_dx, ghost_dy, ghost_dist)

# def simplify_state(state):
#     px, py = state["pacman_position"]

#     # Convert pellet grid back to coordinates
#     pellets = state["pellet_positions"]
#     grid_w = int(np.sqrt(len(pellets)))  # assumes square grid

#     pellet_coords = [
#         (i // grid_w, i % grid_w)
#         for i, v in enumerate(pellets) if v == 1
#     ]

#     # Ghost positions (dict → list of tuples)
#     ghosts = list(state["ghost_positions"].values())

#     # distance to nearest pellet
#     if pellet_coords:
#         nearest_pellet = min(
#             pellet_coords,
#             key=lambda p: abs(px - p[0]) + abs(py - p[1])
#         )
#         food_dx = nearest_pellet[0] - px
#         food_dy = nearest_pellet[1] - py
#     else:
#         food_dx, food_dy = 0, 0

#     # distance to nearest ghost
#     if ghosts:
#         nearest_ghost = min(
#             ghosts,
#             key=lambda g: abs(px - g[0]) + abs(py - g[1])
#         )
#         ghost_dx = nearest_ghost[0] - px
#         ghost_dy = nearest_ghost[1] - py
#     else:
#         ghost_dx, ghost_dy = 0, 0

#     return (
#         px, py,
#         food_dx, food_dy,
#         ghost_dx, ghost_dy
#     )
