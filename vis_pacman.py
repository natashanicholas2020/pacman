import pygame
from env.pacman_env import PacmanEnv

pygame.init()

env = PacmanEnv(render_mode="human")
env.render()

done = False

key_to_action = {
    pygame.K_UP: 0, 
    pygame.K_DOWN: 1, 
    pygame.K_LEFT: 2, 
    pygame.K_RIGHT: 3}

while not done:
    action = None
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key in key_to_action:
                action = key_to_action[event.key]
    
    if action is not None:
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

env.close()
pygame.quit()
