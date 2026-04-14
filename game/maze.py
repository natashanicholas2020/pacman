"""
maze.py — three maze designs for Pac-Man Q-learning

All mazes are 15 rows × 20 cols and satisfy the ghost-pen system constraints:
  - Outer border is all walls
  - Pac-Man starts at (1, 1)
  - Ghost pen cells (7,8)–(7,11) are open (value 0)
  - Pen exit (5,9) is open
  - Every non-wall cell is reachable from (1,1)

Switch between mazes by changing the `maze = ...` line at the bottom.

Legend
------
  0  open floor
  1  wall
  2  pellet (open floor with a collectible)
"""

import pygame
import numpy as np

CELL_SIZE = 50

# ── Maze 1: Grid City ─────────────────────────────────────────────────────
# Wide, regular grid of corridors with two long horizontal highways (rows 5
# and 9) and two long horizontal paths on rows 3 and 11.  Lots of straight
# runs — ideal for early training since the agent can charge down corridors
# without getting blocked.  Pellets are placed at corridor intersections so
# the agent has clear directional cues.  20 pellets.
MAZE_GRID_CITY = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,2,0,0,0,2,0,0,2,0,0,0,2,0,0,2,1],
    [1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1],
    [1,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,1],
    [1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,0,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  # pen row — (7,8-11) are open
    [1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,0,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
    [1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,0,1],
    [1,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,1],
    [1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1],
    [1,2,0,0,2,0,0,0,2,0,0,2,0,0,0,2,0,0,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
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

RANDOM_MAZE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,1],
    [1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],
    [1,0,0,2,0,0,2,0,0,0,0,0,2,0,0,0,0,2,0,1],
    [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,2,0,0,2,0,0,1],
    [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
    [1,2,0,0,0,0,0,2,0,0,2,0,0,0,0,0,2,0,0,1],
    [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])



# ── Active maze ───────────────────────────────────────────────────────────
# Change this line to switch between designs:
#   maze = MAZE_GRID_CITY
#   maze = MAZE_RINGS
#   maze = MAZE_CATACOMBS
maze = MAZE_GRID_CITY

height, width = maze.shape
print(f"Maze: {width}×{height}  |  pellets: {int((maze == 2).sum())}")


# ── Window creation ───────────────────────────────────────────────────────
def create_window():
    pygame.init()
    screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
    pygame.display.set_caption("Pac-Man")
    return screen


# ── Drawing ───────────────────────────────────────────────────────────────
def draw_maze(screen, draw_pellets=True):
    for r in range(height):
        for c in range(width):
            x = c * CELL_SIZE
            y = r * CELL_SIZE

            if maze[r, c] == 1:
                pygame.draw.rect(screen, (0, 0, 200), (x, y, CELL_SIZE, CELL_SIZE))
                # subtle inner highlight for depth
                pygame.draw.rect(screen, (20, 20, 220), (x+1, y+1, CELL_SIZE-2, CELL_SIZE-2))
            elif draw_pellets and maze[r, c] == 2:
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.circle(screen, (255, 120, 120), rect.center, CELL_SIZE // 6)