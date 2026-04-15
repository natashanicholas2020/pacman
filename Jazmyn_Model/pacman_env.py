import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from game.maze import maze as MAZE


class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode="Human"):
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
        # self.max_steps = 750
        self.steps = 0

        # Action Space
        self.actions = {
            0: (-1, 0), # up
            1: (1, 0),  # down
            2: (0, -1), # left
            3: (0, 1)   # right
        }
        self.action_space = spaces.Discrete(len(self.actions))

        self.box_history = []
        self.box_limit = 6
        self.pos_history = []
        self.pos_limit = 8

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
            'pellet': 100,
            'ghost': -500,
            'empty': -0.1,
            'closer_to_pellet': 10,
            'further_from_pellet': -2,
            'close_to_ghost': -5,
            'further_from_ghost': 2,
            'danger_zone': -20,
            'loitering': -6,
            'win': 2000,
            'oob': -5
        }

        

        self.reset()

    def reset(self):

        # Reset step counter
        self.steps = 0
        self.box_history = []
        self.pos_history = []
        self.current_direction = 3
        self.queued_direction = 3
        # Initialize positions
        self.pacman_position = (1, 1)
        self.pellets = set(zip(*np.where(MAZE == 2)))
        self.ghost_positions = {
        ghost: (7, 8 + i)
        for i, ghost in enumerate(self.ghosts)}

        self.ghost_directions = {
            ghost: random.choice(list(self.actions.values()))
            for ghost in self.ghosts
        }

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

        self.steps += 1
        reward = self.rewards["empty"]
        terminated = False

        # move pacman based ONLY on action
        old_pos = self.pacman_position

        dx, dy = self.actions[action]
        nx, ny = old_pos[0] + dx, old_pos[1] + dy

        if (nx, ny) not in self.walls:
            self.pacman_position = (nx, ny)
        else:
            reward += self.rewards["oob"]

        new_pos = self.pacman_position

        # loitering penalty
        box = (self.pacman_position[0] // 2, self.pacman_position[1] // 2)
        self.box_history.append(box)
        if len(self.box_history) > self.box_limit:
            self.box_history.pop(0)
        if len(self.box_history) == self.box_limit and len(set(self.box_history)) <= 2:
            reward += self.rewards["loitering"]

        self.pos_history.append(self.pacman_position)
        if len(self.pos_history) > self.pos_limit:
            self.pos_history.pop(0)
        if len(self.pos_history) == self.pos_limit and len(set(self.pos_history)) <= 3:
            reward += self.rewards["loitering"]

        # if moving closer to nearest pellet
        if self.pellets:
            nearest_pellet = min(
                self.pellets,
                key=lambda p: abs(self.pacman_position[0] - p[0]) + abs(self.pacman_position[1] - p[1])
            )
            old_dist = abs(old_pos[0] - nearest_pellet[0]) + abs(old_pos[1] - nearest_pellet[1])
            new_dist = abs(new_pos[0] - nearest_pellet[0]) + abs(new_pos[1] - nearest_pellet[1])

            if new_dist < old_dist:
                reward += self.rewards["closer_to_pellet"]
            elif new_dist > old_dist:
                reward += self.rewards["further_from_pellet"]

        # pellet
        if self.pacman_position in self.pellets:
            scaling = np.exp(-0.05 * len(self.pellets))
            self.pellets.remove(self.pacman_position)
            reward += (self.rewards["pellet"] * scaling)

        # ghost proximity
        nd = min(abs(self.pacman_position[0] - gpos[0]) 
                 + abs(self.pacman_position[1] - gpos[1]) for gpos in self.ghost_positions.values())
        od = min(abs(old_pos[0] - gpos[0]) 
                 + abs(old_pos[1] - gpos[1]) for gpos in self.ghost_positions.values())
        if nd < od:
            reward += self.rewards["close_to_ghost"]
        elif nd > od:
            reward += self.rewards["further_from_ghost"]
        if nd <= 3:
            reward += self.rewards["danger_zone"]


        # ghosts
        self.ghost_timer += 1
        if self.ghost_timer >= self.ghost_interval:
            self.move_ghosts()
            self.ghost_timer = 0

        # collision
        if self.pacman_position in self.ghost_positions.values():
            reward += self.rewards["ghost"]
            terminated = True

        # win
        if len(self.pellets) == 0:
            reward += self.rewards["win"]
            terminated = True

        if self.steps >= self.max_steps:
            terminated = True

        return self.get_observation(), reward, terminated, {}

    def move_ghosts(self):
        for ghost in self.ghost_positions:
            gx, gy = self.ghost_positions[ghost]
            dx, dy = self.ghost_directions[ghost]

            nx, ny = gx + dx, gy + dy

            # If current direction is blocked → must choose new
            if not (
                0 <= nx < self.grid_size[0]
                and 0 <= ny < self.grid_size[1]
                and (nx, ny) not in self.walls
            ):
                dx, dy = self.choose_new_direction(ghost, gx, gy, force_turn=True)

            else:
                # Occasionally turn
                if random.random() < 0.2:  # 20% chance to turn
                    dx, dy = self.choose_new_direction(ghost, gx, gy, force_turn=False)

            # Apply movement
            nx, ny = gx + dx, gy + dy
            self.ghost_positions[ghost] = (nx, ny)
            self.ghost_directions[ghost] = (dx, dy)

    def render(self, mode='human'):
        if not hasattr(self, 'renderer'):
            from game.renderer import Renderer
            self.renderer = Renderer()
        self.renderer.render(self)

    def choose_new_direction(self, ghost, gx, gy, force_turn=False):
        current_dx, current_dy = self.ghost_directions[ghost]
        opposite = (-current_dx, -current_dy)

        valid_dirs = []

        for dx, dy in self.actions.values():
            nx, ny = gx + dx, gy + dy

            if (
                0 <= nx < self.grid_size[0]
                and 0 <= ny < self.grid_size[1]
                and (nx, ny) not in self.walls
            ):
                valid_dirs.append((dx, dy))

        if not valid_dirs:
            return current_dx, current_dy  # stuck (shouldn’t happen)

        # Avoid reversing unless forced
        if not force_turn:
            valid_dirs = [d for d in valid_dirs if d != opposite] or valid_dirs

        # Bias toward continuing straight
        weights = []
        for d in valid_dirs:
            if d == (current_dx, current_dy):
                weights.append(3.0)  # prefer straight
            else:
                weights.append(1.0)

        return random.choices(valid_dirs, weights=weights)[0]