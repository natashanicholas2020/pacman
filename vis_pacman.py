import pygame
from env.pacman_env import PacmanEnv

pygame.init()

env = PacmanEnv(render_mode="human")
env.render()

clock = pygame.time.Clock()

direction = 3 # start moving right
running = True


key_to_action = {
    pygame.K_UP: 0, 
    pygame.K_DOWN: 1, 
    pygame.K_LEFT: 2, 
    pygame.K_RIGHT: 3}

while running:
    action = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = True

        keys = pygame.key.get_pressed()

        for key, action in key_to_action.items():
            if keys[key]:
                env.queued_direction = action

    obs, reward, done, info = env.step(direction) 

    if done:
        running = False

    clock.tick(3)

env.close()
pygame.quit()
