"""
q_learning.py

State representation and Q-learning tuned for the ghost-pen environment.
The key insight: track number of ACTIVE ghosts as part of state so the
agent learns different policies for "1 ghost" vs "4 ghosts" situations.
"""

import random
from collections import defaultdict
import numpy as np


def Q_learning(env, num_episodes=8000, gamma=0.95, epsilon=1.0,
               decay_rate=0.9997, alpha=0.2):

    Q = {}
    Q_update_counts = {}

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
        state         = simplify_state(obs)
        done          = False
        episode_reward = 0
        steps          = 0

        while not done:
            if state not in Q:
                Q[state] = np.zeros(env.action_space.n)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
                # q_vals = [Q[(state, a)] for a in actions]
                # max_q  = max(q_vals)
                # # random tie-breaking prevents direction lock-in
                # action = random.choice([a for a, v in zip(actions, q_vals) if v == max_q])

            next_obs, reward, done, _ = env.step(action)
            next_state     = simplify_state(next_obs)

            if next_state not in Q:
                Q[next_state] = np.zeros(env.action_space.n)

            # Update Q-value using the Q-learning update rule
            if (state, action) not in Q_update_counts:
                Q_update_counts[(state, action)] = 0

            Q_update_counts[(state, action)] += 1

            alpha = 1 / (1 + Q_update_counts.get((state, action)))

            Q[state][action] += alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state][action]
                )

            episode_reward += reward
            steps          += 1

            state = next_state

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

    # ── final summary ──────────────────────────────────────────────
    total_wins = sum(metrics["win_flags"])
    print("\n" + "=" * 60)
    print(f"Training complete  ({num_episodes} episodes)")
    print(f"Total wins         : {total_wins}  ({total_wins/num_episodes*100:.1f}%)")
    print(f"Last 100 win%      : {metrics['win_pct_100'][-1]}%")
    print(f"Last 500 win%      : {metrics['win_pct_500'][-1]}%")
    print(f"Avg reward (last 500): {round(float(np.mean(metrics['episode_rewards'][-500:])), 1)}")
    print("=" * 60)

    return Q, metrics


# ── State representation ───────────────────────────────────────────────────
def simplify_state(state):
    px, py       = state["pacman_position"]
    pellets_flat = state["pellet_positions"]
    ghost_dict   = state["ghost_positions"]   # {(name,color): (r,c)}

    # rebuild pellet coords
    total  = len(pellets_flat)
    grid_w = int(round(total ** 0.5))
    pellet_coords = [
        (i // grid_w, i % grid_w)
        for i, v in enumerate(pellets_flat) if v == 1
    ]

    # separate active ghosts (those NOT sitting in the pen rows 7)
    # active means they've been released; we detect this by checking if
    # their position is outside the pen area
    PEN_ROWS = {7}  # pen is on row 7
    all_ghost_pos = list(ghost_dict.values())
    active_ghosts = [g for g in all_ghost_pos if g[0] not in PEN_ROWS]
    # fallback: if none classified as active, use all
    if not active_ghosts:
        active_ghosts = all_ghost_pos

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ── nearest pellet ─────────────────────────────────────────────
    if pellet_coords:
        sorted_p = sorted(pellet_coords, key=lambda p: manhattan((px, py), p))
        np1 = sorted_p[0]
        pdx   = int(np.sign(np1[0] - px))
        pdy   = int(np.sign(np1[1] - py))
        pdist = min(manhattan((px, py), np1), 8)
        # second-nearest lookahead
        np2     = sorted_p[1] if len(sorted_p) > 1 else np1
        pdx2    = int(np.sign(np2[0] - px))
        pdy2    = int(np.sign(np2[1] - py))
    else:
        pdx = pdy = pdist = pdx2 = pdy2 = 0

    # ── nearest active ghost ───────────────────────────────────────
    if active_ghosts:
        nearest_g = min(active_ghosts, key=lambda g: manhattan((px, py), g))
        gdx    = int(np.sign(nearest_g[0] - px))
        gdy    = int(np.sign(nearest_g[1] - py))
        gdist  = min(manhattan((px, py), nearest_g), 8)
        danger = int(gdist <= 3)
        n_active = min(len(active_ghosts), 4)
    else:
        gdx = gdy = danger = n_active = 0
        gdist = 8

    # ── wall awareness (can I move in each direction?) ─────────────
    actions_deltas = [(-1,0),(1,0),(0,-1),(0,1)]
    walls_near = tuple(
        int((px + dr, py + dc) in _get_walls(state))
        for dr, dc in actions_deltas
    )

    return (
        pdx, pdy, min(pdist, 6),   # nearest pellet direction + dist (capped)
        pdx2, pdy2,                # second-nearest pellet lookahead
        gdx, gdy, min(gdist, 6),   # nearest ghost direction + dist
        danger,                    # 1 if ghost is ≤ 3 steps away
        n_active,                  # how many ghosts are currently out
        walls_near,                # (up_wall, down_wall, left_wall, right_wall)
    )


# cache for wall set (rebuilt lazily from observation — walls never change)
_wall_cache = None

def _get_walls(state):
    """Return the wall set, computed once from the MAZE module."""
    global _wall_cache
    if _wall_cache is None:
        try:
            from game.maze import maze as MAZE
            import numpy as _np
            _wall_cache = set(map(tuple, zip(*_np.where(MAZE == 1))))
        except Exception:
            _wall_cache = set()
    return _wall_cache

