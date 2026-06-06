# 2_1_MADQN: Cooperative Multi-Agent DQN

This experiment extends `2_1_DQN` into a cooperative two-UAV waypoint task.
The original single-agent DQN folder is kept unchanged as the baseline.

## Environment

- 2D grid: `20x20`
- Agents: 4 UAVs, default starts at the four map corners
- Action space: `MultiDiscrete([4, 4, 4, 4])`
- Mission: all UAVs share ten unordered waypoints and complete coverage together
- Waypoint reward: only the first UAV visit to each waypoint receives the waypoint bonus
- Collision rules: wall, obstacle, same-cell entry, and swap collisions block movement and add penalties

Each UAV receives one local observation row. The environment returns
`obs_batch.shape == (n_agents, obs_dim)`.

Observation contents per UAV:

- own normalized position and energy
- global waypoint visited flags
- adjacent blocked flags
- direction to the nearest unvisited waypoint for that UAV
- one-hot agent id
- relative positions of the other UAVs

## Learning

MADQN uses one shared Q-network for all UAVs.

- Shared network: `state -> Q(action)` with 4 discrete actions
- Replay buffer: one shared buffer with one transition per UAV per joint step
- Exploration: epsilon-greedy shared by all agents
- Target network: periodic hard update

Current harder-task defaults:

- episodes: `10000`
- epsilon: `1.0 -> 0.05`, decay `0.9992`
- replay buffer: `600000`
- batch size: `256`
- target update: `1500` learning steps
- hidden dim: `512`
- max steps: `1000`

Changing from the older 2-UAV / 5-waypoint task changes the observation
dimension, so old checkpoints must be retrained.

## Run

From the repository root:

```powershell
cd ".\Multi-Agent\2_1_MADQN"
..\..\venv\Scripts\python.exe .\train.py
```

Regenerate plots from an existing checkpoint:

```powershell
cd ".\Multi-Agent\2_1_MADQN"
..\..\venv\Scripts\python.exe .\test.py
```

## Results

Training writes these files under `results/`:

- `training_log.csv`
- `madqn_model.pt`
- `training_curve.png`
- `success_curve.png`
- `loss_curve.png`
- `best_path.png`
- `flight.gif`
