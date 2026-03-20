# This is where we should render everything
import pygame
import game.maze as maze
import game.sprites as sprites

screen = maze.create_window()
clock = pygame.time.Clock()

# pacman position
pacman_pos = [1, 1]

# ghost positions and colors
ghosts = [
    {"pos": [7, 8], "color": (255, 0, 0)},
    {"pos": [7, 9], "color": (255, 184, 255)},
    {"pos": [7, 10], "color": (0, 255, 255)},
    {"pos": [7, 11], "color": (255, 184, 82)}
]

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

    # draw maze, pacman, and ghosts
    maze.draw_maze(screen)

    # draw sprites
    sprites.draw_pacman(screen, pacman_pos, mouth_direction)

    for ghost in ghosts:
        sprites.draw_ghost(screen, ghost["pos"], ghost["color"])

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
