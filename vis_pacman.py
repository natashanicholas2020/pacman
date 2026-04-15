"""
vis_pacman.py  –  train then visualise a greedy episode
Run:  python vis_pacman.py
"""
import pickle
import sys
import pygame
import numpy as np
from env.pacman_env import PacmanEnv
from q_learning import Q_learning, simplify_state

train_flag = "train" in sys.argv
gui_flag = "gui" in sys.argv

filename = "MAZE-CATACOMBS-5000.pickle"


# ––––––– Plotting metrics –––––––––––––––––––––––––––––––––

def plot_rewards(rewards, filename="reward_plot.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(rewards)), rewards, s=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.grid()
    plt.savefig(filename)
    plt.close()

# ── Train ────────────────────────────────────────────────────────────────────
if train_flag:
    print("Training …")
    train_env = PacmanEnv(render_mode=None)
    Q, metrics = Q_learning(
        train_env,
        num_episodes=5000,
        gamma=0.85,
        epsilon=1.0,
        decay_rate=0.999,
        alpha=0.1,
    )
    train_env.close()
    # Save the Q-table dict to a file
    
    with open(filename, "wb") as handle:
        pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plot_rewards(metrics["episode_rewards"], filename="reward_plot.png")

# -- Softmax exploration (1000 episodes) ───────────────────────────────────────────
def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)

print("\nEvaluating (1000 softmax episodes) …")

soft_env = PacmanEnv(render_mode=None)
soft_wins = 0
soft_rewards = []

Q_table = np.load(filename, allow_pickle=True)

for _ in range(1000):
    obs, _, _, _ = soft_env.reset()
    s = simplify_state(obs)
    total_r = 0
    done = False
    while not done:

        s = simplify_state(obs)
        try:
            action = np.random.choice(
                soft_env.action_space.n, p=softmax(Q_table[s])
            )  # Select action using softmax over Q-values
        except KeyError:
            action = (
                soft_env.action_space.sample()
                )  # Fallback to random action if state not in Q-table
        
        obs, r, done, _ = soft_env.step(action)
        s = simplify_state(obs)
        total_r += r
    soft_rewards.append(total_r)

    if sum(obs["pellet_positions"]) == 0:
        soft_wins += 1

soft_env.close()
print(f"Softmax win rate : {soft_wins}/1000  ({soft_wins / 1000 * 100}%)")
print(f"Avg reward      : {round(np.mean(soft_rewards), 1)}")


# ── Visual episode ────────────────────────────────────────────────────────────
pygame.init()
env = PacmanEnv(render_mode="Human")
obs, _, _, _ = env.reset()
s = simplify_state(obs)
total_reward = 0
env.render()

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    try:
        action = np.random.choice(
            soft_env.action_space.n, p=softmax(Q_table[s])
        )  # Select action using softmax over Q-values
    except KeyError:
        action = (
            soft_env.action_space.sample()
            ) 
    # action = best_action(s, Q)  # Greedy action (no exploration)
    obs, reward, done, _ = env.step(action)
    s = simplify_state(obs)
    env.render()
    total_reward += reward

    if done:
        remaining = sum(obs["pellet_positions"])
        result = "WIN" if remaining == 0 else "LOSS"
        print(f"\nVisual episode: {result}  |  reward = {total_reward:.1f}")
        running = False

    clock.tick(5)

env.close()
pygame.quit()


