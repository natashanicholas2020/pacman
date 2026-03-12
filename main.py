# This is where we should render everything
import pygame
import maze

screen = maze.create_window()
clock = pygame.time.Clock()

pacman_pos = [1, 1]
mouth_direction = 3

# change mouth direction based on movement
running = True
while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_UP:
                mouth_direction = 0
            elif event.key == pygame.K_DOWN:
                mouth_direction = 1
            elif event.key == pygame.K_LEFT:
                mouth_direction = 2
            elif event.key == pygame.K_RIGHT:
                mouth_direction = 3

    screen.fill((0, 0, 0))

    # draw maze and pacman
    maze.draw_maze(screen)
    maze.draw_pacman(screen, pacman_pos, mouth_direction)

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
