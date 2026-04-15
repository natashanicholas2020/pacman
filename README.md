Kaitlyn Freeley, Jazmyn Harris, Natasha Nicholas

Pac-Man Q-Learning Project
A reinforcement learning project where agents learn to play Pac-Man using Q-learning across multiple custom environments. The project supports training, evaluation, and real-time visualization using pygame.

Project Structure
.
├── vis_pacman.py # Main entry point (train, eval, GUI)
├── q_learning.py # Q-learning implementation
├── Jazmyn_Model/
│ └── pacman_env.py # Jazmyn's environment
├── Kaitlyn_Model/
│ └── pacman_env.py # Kaitlyn's environment
├── Natasha_Model/
│ └── pacman_env.py # Natasha's environment
├── pickle_files/ # Saved Q-tables
│ └── \*.pickle
├── reward_plot.png # (Optional) training visualization
└── README.md

Setup

1. Clone the Repository
   git clone <your-repo-url>
   cd <your-repo-folder>

2. Install Dependencies
   pip install numpy pygame matplotlib

Running the Program

The main script is:
python vis_pacman.py [mode flags] [environment flag]

Environment Selection (Required)
You must specify exactly one of the following:
jazmyn
natasha
kaitlyn
Example:
python vis_pacman.py train jazmyn

Modes:

Training
Train a Q-learning agent:
python vis*pacman.py train jazmyn
Runs 5000 episodes
Saves Q-table to:
pickle_files/train_Q_table*<name>.pickle

Evaluation
Evaluate a trained agent using softmax exploration:
python vis_pacman.py eval jazmyn
Outputs:
Win rate (out of 1000 episodes)
Average reward

GUI Visualization

GUI requires a trained/evaluated model
python vis_pacman.py gui <name>
Runs evaluation first
Then launches a visual episode using pygame
Agent acts using learned Q-values

Recommended Workflows

# 1. Train an agent yourself

python vis_pacman.py train eval gui <name>

# 2. Evaluate and visualize our models

python vis_pacman.py eval gui <name>

OPTIONAL:

# 1. Change the maze

- In game/maze.py, change the variable "maze" (line 97) to a different name ["ORIGINAL_MAZE", "MAZE_CATACOMBS", "MAZE_RINGS", "GRID_CITY"]
  The default maze is GRID_CITY

# 2. Change the number of episodes

- In vis_pacman.py, change num_episodes=5000 (line 66) to a different number
  The default value is 5000

# 3. Change the number of evaluation episodes

- In vis_pacman.py, change eval_episodes = 500 (line 87) to a different number
  The default value is 500

Features
Q-learning with:
Epsilon decay
Learning rate tuning
Discount factor control
Softmax-based evaluation policy
Multiple interchangeable environments
Real-time visualization with pygame
Reward tracking (optional plotting)

Common Errors
No environment selected
Error: No environment flag set.
Multiple environments selected
Error: Multiple environment flags set.
