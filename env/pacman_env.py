import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from game.maze import maze as MAZE


class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Define the maze and map
        self.grid_size = MAZE.shape
        self.walls = set(zip(*np.where(MAZE == 1)))
        self.pellets = set(zip(*np.where(MAZE == 2)))
        self.ghosts = [
            ("Blinky", (255, 0, 0)),
            ("Pinky", (255, 184, 255)),
            ("Inky", (0, 255, 255)),
            ("Clyde", (255, 184, 82)),
        ]
        self.ghost_timer = 0
        self.ghost_interval = 3 # move ghosts every 3 steps

        # Episode control
        self.max_steps = 1000
        self.steps = 0

        # Action Space
        self.actions = {
            0: (-1, 0), # up
            1: (1, 0),  # down
            2: (0, -1), # left
            3: (0, 1)   # right
        }
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space
        obs_space_dict = {
            "pacman_position": spaces.Tuple(
                (spaces.Discrete(self.grid_size[0]), spaces.Discrete(self.grid_size[1]))
            ),
            "pellet_positions": spaces.MultiBinary(
                self.grid_size[0] * self.grid_size[1]
            ),
            "ghost_positions": spaces.Dict(
                {
                    ghost: spaces.Tuple(
                        (
                            spaces.Discrete(self.grid_size[0]),
                            spaces.Discrete(self.grid_size[1]),
                        )
                    )
                    for ghost in self.ghosts
                }
            ),
        }

        self.observation_space = spaces.Dict(obs_space_dict)

        # Rewards
        self.rewards = {
            'pellet': 10,
            'ghost': -100,
            'empty': -1,
            'win': 500,
            'oob': -10
        }

        self.reset()

    def reset(self):

        # Reset step counter
        self.steps = 0

        self.current_direction = 3
        self.queued_direction = 3
        # Initialize positions
        self.pacman_position = (1, 1)
        self.pellets = set(zip(*np.where(MAZE == 2)))
        self.ghost_positions = {
        ghost: (7, 8 + i)
        for i, ghost in enumerate(self.ghosts)}

        return self.get_observation(), 0, False, {}

    def get_observation(self):
        pellet_grid = np.zeros(self.grid_size[0] * self.grid_size[1], dtype=np.int8)

        for (x, y) in self.pellets:
            idx = x * self.grid_size[1] + y
            pellet_grid[idx] = 1

        return {
            "pacman_position": self.pacman_position,
            "pellet_positions": pellet_grid,
            "ghost_positions": self.ghost_positions
        }

    def step(self, action):

        ## This is a fix for gym environment.
        if isinstance(action, str):
            action = self.actions.index(action)

        # Increment step counter
        self.steps += 1

        # Move pacman
        # reverse option
        opposite = (self.current_direction + 2) % 4
        if self.queued_direction == opposite:
            self.current_direction = self.queued_direction
            
        qd = self.queued_direction
        dx, dy = self.actions[qd]
        nx = self.pacman_position[0] + dx
        ny = self.pacman_position[1] + dy

        if (nx, ny) not in self.walls:
            self.current_direction = qd

        dx, dy = self.actions[self.current_direction]
        nx = self.pacman_position[0] + dx
        ny = self.pacman_position[1] + dy

        reward = self.rewards['empty']
        terminated = False

        # wall collision
        if (nx, ny) in self.walls:
            reward = self.rewards['oob']
            nx, ny = self.pacman_position

        self.pacman_position = (nx, ny)

        # pellet
        if self.pacman_position in self.pellets:
            self.pellets.remove(self.pacman_position)
            reward = self.rewards['pellet']

        # Move ghosts
        self.ghost_timer += 1
        if self.ghost_timer >= self.ghost_interval:
            self.move_ghosts()
            self.ghost_timer = 0

        # ghost collision
        if self.pacman_position in self.ghost_positions.values():
            reward = self.rewards['ghost']
            terminated = True

        # win condition
        if len(self.pellets) == 0:
            reward = self.rewards['win']
            terminated = True

        # Truncate episode if max steps reached and not already done
        truncated = False
        if not terminated and self.steps >= self.max_steps:
            terminated = True
            truncated = True

        if self.render_mode == 'human':
            self.render()

        return self.get_observation(), reward, terminated, {}

    # fix the ghost movement to be more deterministic and less random. Continue moving in the same direction until hitting a wall, then choose a new direction.
    def move_ghosts(self):
        for ghost in self.ghost_positions:
            gx, gy = self.ghost_positions[ghost]

            possible_moves = [
                (gx + dx, gy + dy)
                for dx, dy in self.actions.values()
            ]

            valid_moves = [
                move for move in possible_moves
                if move not in self.walls
            ]

            if valid_moves:
                self.ghost_positions[ghost] = random.choice(valid_moves)

    def render(self, mode='human'):
        if not hasattr(self, 'renderer'):
            from game.renderer import Renderer
            self.renderer = Renderer()
        self.renderer.render(self)
