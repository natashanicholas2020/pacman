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
            'empty': -1,
            'closer': 1,
            'further': -1,
            'win': 2000,
            'oob': -5
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
        self.total_pellets = len(self.pellets)
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

        # if moving closer to nearest pellet
        if self.pellets:
            nearest_pellet = min(
                self.pellets,
                key=lambda p: abs(self.pacman_position[0] - p[0]) + abs(self.pacman_position[1] - p[1])
            )
            old_dist = abs(old_pos[0] - nearest_pellet[0]) + abs(old_pos[1] - nearest_pellet[1])
            new_dist = abs(new_pos[0] - nearest_pellet[0]) + abs(new_pos[1] - nearest_pellet[1])

            if new_dist < old_dist:
                reward += self.rewards["closer"]
            elif new_dist > old_dist:
                reward += self.rewards["further"]

        # pellet
        if self.pacman_position in self.pellets:
            self.pellets.remove(self.pacman_position)
            reward += self.rewards["pellet"]
        
        # progress reward (encourages clearing maze)
        reward += -1 * len(self.pellets) * 0.01

        if self.pacman_position in self.ghost_positions.values():
            reward += self.rewards["ghost"]
            terminated = True

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
    
    # def move_ghosts(self):
    #     for ghost in self.ghost_positions:
    #         gx, gy = self.ghost_positions[ghost]
    #         dx, dy = self.ghost_directions[ghost]

    #         def is_valid(x, y):
    #             return (
    #                 0 <= x < self.grid_size[0]
    #                 and 0 <= y < self.grid_size[1]
    #                 and (x, y) not in self.walls
    #             )

    #         # -----------------------------------------
    #         # 1. TRY TO CONTINUE CURRENT DIRECTION
    #         # -----------------------------------------
    #         nx, ny = gx + dx, gy + dy

    #         can_continue = is_valid(nx, ny)

    #         # Optional randomness: occasionally force a turn
    #         force_turn = random.random() < 0.15  # tweak 0.05–0.3

    #         if not can_continue or force_turn:
    #             # -----------------------------------------
    #             # 2. PICK NEW RANDOM VALID DIRECTION
    #             # -----------------------------------------
    #             possible_dirs = []

    #             for ndx, ndy in self.actions.values():
    #                 tx, ty = gx + ndx, gy + ndy
    #                 if is_valid(tx, ty):
    #                     possible_dirs.append((ndx, ndy))

    #             if possible_dirs:
    #                 dx, dy = random.choice(possible_dirs)
    #                 nx, ny = gx + dx, gy + dy
    #             else:
    #                 # trapped (rare)
    #                 dx, dy = 0, 0
    #                 nx, ny = gx, gy

    #         # -----------------------------------------
    #         # 3. APPLY MOVE
    #         # -----------------------------------------
    #         self.ghost_positions[ghost] = (nx, ny)
    #         self.ghost_directions[ghost] = (dx, dy)

    # ── ghost movement ────────────────────────────────────────────────
    # def move_ghosts(self):
        
    #     for ghost in self.ghost_positions:
    #         gx, gy = self.ghost_positions[ghost]
    #         dx, dy = self.ghost_directions[ghost]

    #         possible_moves = [
    #                 (gx + mx, gy + my)
    #                 for mx, my in self.actions.values()
    #             ]

    #         valid_moves = [
    #             move for move in possible_moves
    #             if (0 <= move[0] < self.grid_size[0]
    #                 and 0 <= move[1] < self.grid_size[1]
    #                 and move not in self.walls)
    #         ]

    #         # Opposite direction check to prevent ghosts from immediately reversing direction
    #         opposite = (-gx, -gy)

    #         if len(valid_moves) > 1 and opposite in valid_moves:
    #             valid_moves.remove(opposite)
            
    #         if len(valid_moves) > 2:
    #             if (dx, dy) in valid_moves:
    #                 choices = [(dx, dy)] * 2 + [d for d in valid_moves if d != (dx, dy)]
    #                 nx, ny = random.choice(choices)
    #             else:
    #                 nx, ny = random.choice(valid_moves)
    #         else:
    #             nx, ny = gx + dx, gy + dy

    #         # Check wall OR out-of-bounds
    #         if not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]) or (nx, ny) in self.walls:
    #             if valid_moves:
    #                 move = random.choice(valid_moves)
    #                 self.ghost_positions[ghost] = move
    #                 self.ghost_directions[ghost] = (move[0] - gx, move[1] - gy)

    #             continue

    #         # Normal move
    #         self.ghost_positions[ghost] = (nx, ny)


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
                # Occasionally turn even if not blocked (adds randomness)
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