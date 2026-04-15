import random
from collections import defaultdict
import numpy as np
from collections import deque
from game.maze import maze

def Q_learning(env, num_episodes=5000, gamma=0.9, epsilon=1.0, decay_rate=0.999, alpha=0.1):

    Q = defaultdict(float)
    actions = [0, 1, 2, 3]  # fixed action space
    num_wins = 0

    metrics = {
        "episode_rewards":  [],
        "win_flags":        [],
        "win_pct_100":      [],
        "win_pct_500":      [],
        "epsilon_log":      [],
        "pellets_eaten":    [],
        "steps_per_episode":[],
    }

    for episode in range(num_episodes):
        obs, _, _, _ = env.reset()
        state = simplify_state(obs) 
        done = False
        episode_reward = 0
        steps = 0

        while not done:

            # ε-greedy
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(actions, key=lambda a: Q[(state, a)])

            next_obs, reward, done, info = env.step(action)
            next_state = simplify_state(next_obs)

            episode_reward += reward
            steps += 1

            # Q-learning update
            if done:
                max_next_Q = 0
            else:
                max_next_Q = max(Q[(next_state, a)] for a in actions)

            Q[(state, action)] += alpha * (
                reward + gamma * max_next_Q - Q[(state, action)]
            )

            if(reward >= 1000):
                num_wins += 1

            state = next_state

        #if(episode % 100 == 0):
            #print(episode, ":", episode_reward)

	# ── win detection ──────────────────────────────────────────
        remaining = int(np.sum(next_obs["pellet_positions"]))
        won = (remaining == 0)

        metrics["episode_rewards"].append(episode_reward)
        metrics["win_flags"].append(int(won))
        metrics["epsilon_log"].append(round(epsilon, 4))
        metrics["pellets_eaten"].append(
            int(sum(1 for v in obs["pellet_positions"] if v == 1)) - remaining
        )
        metrics["steps_per_episode"].append(steps)

        w100 = metrics["win_flags"][max(0, episode - 99):]
        w500 = metrics["win_flags"][max(0, episode - 499):]
        metrics["win_pct_100"].append(round(100 * sum(w100) / len(w100), 2))
        metrics["win_pct_500"].append(round(100 * sum(w500) / len(w500), 2))

        if episode % 100 == 0:
            avg_r    = round(float(np.mean(metrics["episode_rewards"][-100:])), 1)
            win_pct  = metrics["win_pct_100"][-1]
            pellets  = metrics["pellets_eaten"][-1]
            print(
                f"Ep {episode:5d} | reward={episode_reward:8.1f} | "
                f"avg(100)={avg_r:8.1f} | win%={win_pct:5.1f}% | "
                f"ε={epsilon:.4f} | pellets={pellets}"
            )

        epsilon = max(0.05, epsilon * decay_rate)

    total_wins = sum(metrics["win_flags"])
    print("\n" + "=" * 60)
    print(f"Training complete  ({num_episodes} episodes)")
    print(f"Total wins         : {total_wins}  ({total_wins/num_episodes*100:.1f}%)")
    print(f"Last 100 win%      : {metrics['win_pct_100'][-1]}%")
    print(f"Last 500 win%      : {metrics['win_pct_500'][-1]}%")
    print(f"Avg reward (last 500): {round(float(np.mean(metrics['episode_rewards'][-500:])), 1)}")
    print("=" * 60)

    return Q, metrics


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
    #if pellet_coords:
    #    nearest_pellet = min(
    #        pellet_coords,
    #        key=lambda p: abs(px - p[0]) + abs(py - p[1])
    #    )
    #    food_dx = nearest_pellet[0] - px
    #    food_dy = nearest_pellet[1] - py
    #else:
    #    food_dx, food_dy = 0, 0
    nearest_pellet, dist = find_nearest_pellet(
        maze,
        (px, py),
        pellet_coords
    )

    if nearest_pellet:
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

def find_nearest_pellet(maze, start, pellets):
    rows, cols = maze.shape
    visited = set()
    queue = deque([(start, 0)])  # ((x, y), distance)

    while queue:
        (x, y), dist = queue.popleft()

        if (x, y) in visited:
            continue
        visited.add((x, y))

        # found a pellet
        if (x, y) in pellets:
            return (x, y), dist

        # explore neighbors
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy

            if (
                0 <= nx < rows and
                0 <= ny < cols and
                maze[nx][ny] != 1  # not a wall
            ):
                queue.append(((nx, ny), dist + 1))

    return None, None