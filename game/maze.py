# creates the maze
# we should try to make the pellets in a file that addresses the rewards
import pygame
import numpy as np


# ── Maze 1: Grid City ─────────────────────────────────────────────────────
# Wide, regular grid of corridors with two long horizontal highways (rows 5
# and 9) and two long horizontal paths on rows 3 and 11.  Lots of straight
# runs — ideal for early training since the agent can charge down corridors
# without getting blocked.  Pellets are placed at corridor intersections so
# the agent has clear directional cues.  20 pellets.
GRID_CITY = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,2,0,0,2,0,0,0,2,0,0,2,0,0,2,1],
    [1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1],
    [1,0,1,2,0,0,0,0,0,0,0,0,0,0,0,2,1,0,1],
    [1,0,1,0,1,1,1,0,1,1,0,1,1,1,0,0,1,0,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  
    [1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,0,1,0,1,1,1,0,1,1,0,1,1,1,0,0,1,0,1],
    [1,0,1,2,0,0,0,0,0,0,0,0,0,0,0,2,1,0,1],
    [1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1],
    [1,2,0,0,2,0,0,2,0,0,0,2,0,0,2,0,0,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
])

# ── Maze 2: Rings ─────────────────────────────────────────────────────────
# Three concentric loops: an outer perimeter corridor, a middle ring, and an
# inner chamber surrounding the pen.  Rings connect at a small number of
# junction points — these become natural chokepoints the agent must learn to
# navigate.  The structure rewards agents that can plan ahead because dead
# ends force backtracking.  16 pellets.
MAZE_RINGS = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1],
    [1,0,1,2,0,0,0,0,0,2,0,2,0,0,0,0,2,1,0,1],
    [1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1],
    [1,0,1,0,1,2,0,0,0,0,0,0,0,0,2,1,0,1,0,1],
    [1,0,1,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],  # pen row
    [1,0,1,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,0,1],
    [1,0,1,0,1,2,0,0,0,0,0,0,0,0,2,1,0,1,0,1],
    [1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1],
    [1,0,1,2,0,0,0,0,0,2,0,2,0,0,0,0,2,1,0,1],
    [1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
])

# ── Maze 3: Catacombs ─────────────────────────────────────────────────────
# Organic, asymmetric layout of rooms connected by narrow single-cell
# passages.  Several dead-end pockets mean the agent must decide whether to
# commit to a pocket for pellets while risking being cornered.  Vertically
# symmetric — the top and bottom halves mirror each other — which helps the
# Q-table generalise faster.  18 pellets.
MAZE_CATACOMBS = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,0,2,1,0,0,2,0,0,2,1,0,0,0,2,0,1],
    [1,0,1,1,0,0,1,0,1,1,0,1,0,1,0,1,1,0,0,1],
    [1,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,1],
    [1,1,1,0,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,0,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],  # pen row
    [1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,1,1,0,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1],
    [1,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,1],
    [1,0,1,1,0,0,1,0,1,1,0,1,0,1,0,1,1,0,0,1],
    [1,2,0,0,0,2,1,0,0,2,0,0,2,1,0,0,0,2,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
])

ORIGINAL_MAZE = np.array([
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

maze = GRID_CITY

CELL_SIZE = 50
height, width = maze.shape

print(f"Maze dimensions: {width} x {height}")
# creates the game window
def create_window():
    pygame.init()
    screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
    pygame.display.set_caption("Pac-Man")
    return screen


# draws the game background (including pellets, we might want to move the pellets into the rewards file though)
def draw_maze(screen, draw_pellets=True):
    for r in range(height):
        for c in range(width):

            x = c * CELL_SIZE
            y = r * CELL_SIZE

            # walls
            if maze[r, c] == 1:
                pygame.draw.rect(screen, (0, 0, 255), (x, y, CELL_SIZE, CELL_SIZE))

            # pellets
            elif draw_pellets and maze[r, c] == 2:
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.circle(screen, (255, 150, 150), rect.center, CELL_SIZE // 6)
