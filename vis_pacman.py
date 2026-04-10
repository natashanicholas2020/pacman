import pygame
from env.pacman_env import PacmanEnv
from q_learning import Q_learning
from q_learning import simplify_state

pygame.init()

# --- TRAIN ---
train_env = PacmanEnv(render_mode=None)
Q = Q_learning(train_env, num_episodes=5000)
train_env.close()

# --- RUN WITH VISUALS ---
env = PacmanEnv(render_mode="Human")
obs, _, _, _ = env.reset()
state = simplify_state(obs)
total_reward = 0

clock = pygame.time.Clock()
running = True

def best_action(state, Q):
    return max([0,1,2,3], key=lambda a: Q.get((state, a), 0))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = best_action(state, Q)

    obs, reward, done, info = env.step(action)
    state = simplify_state(obs)

    total_reward += reward

    if done:
        running = False
        print("Final evaluation reward:", total_reward)

    clock.tick(5)

env.close()
pygame.quit()