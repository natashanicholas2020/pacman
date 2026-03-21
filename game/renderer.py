import pygame
from game import maze, sprites

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = maze.create_window()
        self.clock = pygame.time.Clock()

    def render(self, env):
        self.screen.fill((0, 0, 0))

        # draw walls only
        maze.draw_maze(self.screen, draw_pellets=False)

        # draw pellets from env state
        for (r, c) in env.pellets:
            x = c * maze.CELL_SIZE
            y = r * maze.CELL_SIZE

            rect = pygame.Rect(x, y, maze.CELL_SIZE, maze.CELL_SIZE)
            pygame.draw.circle(
                self.screen,
                (255, 150, 150),
                rect.center,
                maze.CELL_SIZE // 6
            )

        # draw pacman
        sprites.draw_pacman(self.screen, list(env.pacman_position), 3)

        # draw ghosts
        for (name, color), pos in env.ghost_positions.items():
            sprites.draw_ghost(self.screen, list(pos), color)

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.quit()
