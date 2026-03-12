# creates the maze
# we should try to make the pellets in a file that addresses the rewards
import pygame
import numpy as np


# maze setup
maze = np.array([
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,1],
[1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],
[1,0,0,2,0,0,2,0,0,2,0,0,2,0,0,2,0,0,2,1],
[1,1,1,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1],
[1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1],
[1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1],
[1,2,0,0,2,0,0,1,0,0,0,0,1,0,0,2,0,0,2,1],
[1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,1],
[1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1],
[1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1],
[1,2,0,0,2,0,0,2,0,0,0,0,2,0,0,2,0,0,2,1],
[1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])

CELL_SIZE = 50
height, width = maze.shape


# creates the game window
def create_window():
    pygame.init()
    screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
    pygame.display.set_caption("Pac-Man")
    return screen


# draws the game background (including pellets, we might want to move the pellets into the rewards file though)
def draw_maze(screen):
    for r in range(height):
        for c in range(width):

            x = c * CELL_SIZE
            y = r * CELL_SIZE

            # walls
            if maze[r, c] == 1:
                pygame.draw.rect(screen, (0, 0, 255), (x, y, CELL_SIZE, CELL_SIZE))

            # pellets
            elif maze[r, c] == 2:
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.circle(screen, (255, 150, 150), rect.center, CELL_SIZE//6)