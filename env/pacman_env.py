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
        self.steps = 0

        # Action Space
        self.actions = {
            0: (-1, 0), # up
            1: (1, 0),  # down
            2: (0, -1), # left
            3: (0, 1)   # right
        }
        self.action_space = spaces.Discrete(len(self.actions))

        # State history for loitering penalty
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
        # self.rewards = {
        #     'pellet': 10,
        #     'ghost': -500,
        #     'empty': -1,
        #     'close_to_pellet': 5,
        #     'further_from_pellet': -3,
        #     'close_to_ghost': -20,
        #     'further_from_ghost': 5,
        #     'loitering': -5,
        #     'win': 500,
        #     'oob': -100,
        # } -1172
        self.rewards = {
            'ghost': -1000,
            'win': 1000,

            'pellet': 40,
            'close_to_pellet': 25,
            'further_from_ghost': 2,

            'further_from_pellet': -5,
            'close_to_ghost': -30,

            'empty': -2,
            'loitering': -40,
            
            'oob': -100,
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

        old_pos = self.pacman_position

        # move pacman based ONLY on action
        dx, dy = self.actions[action]
        nx, ny = self.pacman_position[0] + dx, self.pacman_position[1] + dy

        if (nx, ny) not in self.walls:
            self.pacman_position = (nx, ny)
        else:
            reward += self.rewards["oob"]

        # loitering penalty
        box = self.get_box(self.pacman_position)
        self.box_history.append(box)
        if len(self.box_history) > self.box_limit:
            self.box_history.pop(0)
        if len(self.box_history) == self.box_limit:
            if len(set(self.box_history)) <= 2:
                reward += self.rewards["loitering"]


        self.pos_history.append(self.pacman_position)
        if len(self.pos_history) > self.pos_limit:
            self.pos_history.pop(0)
        if len(self.pos_history) == self.pos_limit:
            if len(set(self.pos_history)) <= 3:   # only 3 unique positions in 8 steps
                reward += self.rewards['loitering']
    
        # if moving closer to nearest pellet
        if self.pellets:
            nearest_pellet = min(
                self.pellets,
                key=lambda p: abs(self.pacman_position[0] - p[0]) + abs(self.pacman_position[1] - p[1])
            )
            old_dist = abs(old_pos[0] - nearest_pellet[0]) + abs(old_pos[1] - nearest_pellet[1])
            new_dist = abs(nx - nearest_pellet[0]) + abs(ny - nearest_pellet[1])

            if new_dist < old_dist:
                reward += self.rewards["close_to_pellet"]
            elif new_dist > old_dist:
                reward += self.rewards["further_from_pellet"]

        # if moving closer to nearest ghost
        nearest_ghost = min(
            self.ghost_positions.values(),
            key=lambda g: abs(self.pacman_position[0] - g[0]) + abs(self.pacman_position[1] - g[1])
        )
        old_dist = abs(self.pacman_position[0] - nearest_ghost[0]) + abs(self.pacman_position[1] - nearest_ghost[1])
        new_dist = abs(nx - nearest_ghost[0]) + abs(ny - nearest_ghost[1])

        if new_dist < old_dist:
            reward += self.rewards["close_to_ghost"]
        elif new_dist > old_dist:
            reward += self.rewards["further_from_ghost"]


        # pellet
        if self.pacman_position in self.pellets:
            self.pellets.remove(self.pacman_position)
            reward += self.rewards["pellet"]

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

    # fix the ghost movement to be more deterministic and less random. Continue moving in the same direction until hitting a wall, then choose a new direction.
    def move_ghosts(self):
        
        for ghost in self.ghost_positions:
            gx, gy = self.ghost_positions[ghost]
            dx, dy = self.ghost_directions[ghost]

            possible_moves = [
                    (gx + mx, gy + my)
                    for mx, my in self.actions.values()
                ]

            valid_moves = [
                move for move in possible_moves
                if (0 <= move[0] < self.grid_size[0]
                    and 0 <= move[1] < self.grid_size[1]
                    and move not in self.walls)
            ]

            # Opposite direction check to prevent ghosts from immediately reversing direction
            opposite = (-gx, -gy)

            if len(valid_moves) > 1 and opposite in valid_moves:
                valid_moves.remove(opposite)
            
            if len(valid_moves) > 2:
                if (dx, dy) in valid_moves:
                    choices = [(dx, dy)] * 2 + [d for d in valid_moves if d != (dx, dy)]
                    nx, ny = random.choice(choices)
                else:
                    nx, ny = random.choice(valid_moves)
            else:
                nx, ny = gx + dx, gy + dy

            # Check wall OR out-of-bounds
            if not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]) or (nx, ny) in self.walls:
                if valid_moves:
                    move = random.choice(valid_moves)
                    self.ghost_positions[ghost] = move
                    self.ghost_directions[ghost] = (move[0] - gx, move[1] - gy)

                continue

            # Normal move
            self.ghost_positions[ghost] = (nx, ny)

    def get_box(self, pos, box_size=2):
        x, y = pos
        return (x // box_size, y // box_size)

    def render(self, mode='human'):
        if not hasattr(self, 'renderer'):
            from game.renderer import Renderer
            self.renderer = Renderer()
        self.renderer.render(self)
