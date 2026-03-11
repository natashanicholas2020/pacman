import pygame
import numpy as np
import math

#maze setup
maze = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,1],
    [1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],
    [1,0,0,2,0,0,2,0,0,2,0,0,2,0,0,2,0,0,2,1],
    [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
    [1,2,0,0,2,0,0,2,0,0,2,0,0,2,0,0,2,0,0,1],
    [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
    [1,2,0,0,2,0,0,2,0,0,2,0,0,2,0,0,2,0,0,1],
    [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])

pacman_pos = [1, 1]
CELL_SIZE = 50
height, width = maze.shape

#initialization
pygame.init()
screen = pygame.display.set_mode((width*CELL_SIZE, height*CELL_SIZE))
pygame.display.set_caption("Pac-Man")
clock = pygame.time.Clock()

# mouth variables
mouth_opening = 30  # angle
mouth_direction = 3  # 0 = up, 1 = down, 2 = left, 3 = right

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

    # draw maze
    for r in range(height):
        for c in range(width):
            x, y = c*CELL_SIZE, r*CELL_SIZE

            # walls
            if maze[r, c] == 1:
                pygame.draw.rect(screen, (0, 0, 255), (x, y, CELL_SIZE, CELL_SIZE))

            # pellets
            elif maze[r, c] == 2:
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.circle(screen, (255, 150, 150), rect.center, CELL_SIZE//6)

    # draw pacman
    px = pacman_pos[1]*CELL_SIZE + CELL_SIZE//2
    py = pacman_pos[0]*CELL_SIZE + CELL_SIZE//2
    radius = CELL_SIZE//3

    # body
    pygame.draw.circle(screen, (255, 255, 0), (px, py), radius)

    # mouth
    angle_offset = mouth_opening / 2

    # up
    if mouth_direction == 0:
        start_angle = 270 - angle_offset
        end_angle = 270 + angle_offset

    # down
    elif mouth_direction == 1:
        start_angle = 90 - angle_offset
        end_angle = 90 + angle_offset

    # left
    elif mouth_direction == 2:
        start_angle = 180 - angle_offset
        end_angle = 180 + angle_offset

    # right
    elif mouth_direction == 3:
        start_angle = -angle_offset
        end_angle = angle_offset

    # degree to rad
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)

    # triangle points
    p1 = (px, py)
    p2 = (px + radius * math.cos(start_rad), py + radius * math.sin(start_rad))
    p3 = (px + radius * math.cos(end_rad), py + radius * math.sin(end_rad))

    pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3])

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
