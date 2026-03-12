# we should probably only render the maze background here
# we should make pacman in a file that addresses the sprites
# we should try to make the pellets in a file that addresses the rewards
import pygame
import numpy as np
import math

# maze setup
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


# draws pacman (might want to move this into a file that creates only the sprites, ie. pacman, ghosts)
def draw_pacman(screen, pacman_pos, mouth_direction):

    # location
    px = pacman_pos[1]*CELL_SIZE + CELL_SIZE//2
    py = pacman_pos[0]*CELL_SIZE + CELL_SIZE//2

    # body
    radius = CELL_SIZE//3
    pygame.draw.circle(screen, (255, 255, 0), (px, py), radius)

    # mouth
    angle_offset = 15

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
    else:
        start_angle = -angle_offset
        end_angle = angle_offset

    # degree to radian
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)

    # triangle points
    p1 = (px, py)
    p2 = (px + radius*math.cos(start_rad), py + radius*math.sin(start_rad))
    p3 = (px + radius*math.cos(end_rad), py + radius*math.sin(end_rad))

    pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3])