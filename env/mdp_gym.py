import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from game.maze import maze as MAZE


class PacmanEnv(gym.Env):

    def __init__(self):
        super(PacmanEnv, self).__init__()

        # Define the maze
        self.grid_size = (MAZE.shape[0], MAZE.shape[1])
        self.walls = set(zip(*np.where(MAZE == 1)))
        self.pellets = set(zip(*np.where(MAZE == 2)))

        # Episode control
        self.max_steps = 1000
        self.steps = 0

        # Define ghosts
        self.ghosts = ['Blinky', 'Pinky', 'Inky', 'Clyde']

        # Rewards
        self.rewards = {
            'pellet': 10,
            'ghost': -100,
            'empty': -1,
            'win': 500,
            'oob': -10
        }

        # Action space
        self.actions = ['up', 'down', 'left', 'right']
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space
        obs_space_dict = {
            'pacman_position': spaces.Tuple((spaces.Discrete(self.grid_size[0]), spaces.Discrete(self.grid_size[1]))),
            'pellet_positions': spaces.MultiBinary(self.grid_size[0] * self.grid_size[1]),
            'ghost_positions': spaces.Dict({
                ghost: spaces.Tuple((spaces.Discrete(self.grid_size[0]), spaces.Discrete(self.grid_size[1])))
                for ghost in self.ghosts
            })
        }

        self.observation_space = spaces.Dict(obs_space_dict)

        self.reset()

    def reset(self):

        # Reset step counter
        self.steps = 0

        # Initialize positions
        self.pacman_position = (1, 1)
        self.pellets = set(zip(*np.where(MAZE == 2)))
        self.ghost_positions = {
        ghost: (7, 8 + i)
        for i, ghost in enumerate(self.ghosts)}

        return self.get_observation(), {}

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

        ## Thisis a fix for gym environment.
        if isinstance(action, str):
            action = self.actions.index(action)

        # Increment step counter
        self.steps += 1

        move_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        dx, dy = move_map[action]
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

        self.move_ghosts()

        # ghost collision
        if self.pacman_position in self.ghost_positions.values():
            reward = self.rewards['ghost']
            terminated = True

        # win condition
        if len(self.pellets) == 0:
            reward = self.rewards['win']
            terminated = True

        truncated = self.steps >= self.max_steps

        return self.get_observation(), reward, terminated, truncated, {}

    def move_ghosts(self):
        for ghost in self.ghosts:
            gx, gy = self.ghost_positions[ghost]
            possible_moves = [(gx + dx, gy + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            valid_moves = [move for move in possible_moves if move not in self.walls]
            if valid_moves:
                self.ghost_positions[ghost] = random.choice(valid_moves)
