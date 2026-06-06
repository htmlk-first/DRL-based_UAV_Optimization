# 2_5_MADDPG: Cooperative Multi-Agent DDPG

This experiment keeps the 2D cooperative folder structure from `2_4_MADDQN`
and changes the learner from value-based MADDQN to actor-critic MADDPG.
The task remains a cooperative four-UAV waypoint mission, but each UAV now
executes a continuous 2D movement vector instead of a discrete grid action.

## Environment

- 2D grid: `20x20`
- Agents: 4 UAVs, default starts at the four map corners
- Action space: `Box(-1, 1, shape=(4, 2))`
- Action meaning: each UAV outputs `(dx, dy)`, scaled by `max_step_size`
- Mission: all UAVs share ten unordered waypoints and complete coverage together
- Waypoint reward: a waypoint is visited when any UAV enters `wp_reach_radius`
- Collision rules: wall, obstacle cell, same-location, and swap-like UAV conflicts
- Obstacle collision uses swept line-segment tests against the same centered
  unit squares shown by the path/GIF renderer, preventing corner cutting
  between two valid endpoints

Each UAV receives one local observation row. The environment returns
`obs_batch.shape == (n_agents, obs_dim)`.

Observation contents per UAV:

- own normalized continuous position and energy
- global waypoint visited flags
- adjacent blocked flags
- direction to the nearest unvisited waypoint for that UAV
- one-hot agent id
- relative positions of the other UAVs
- previous continuous action and whether that action was rejected
- direction to a distinct greedily assigned waypoint, avoiding duplicate goals

## Learning

MADDPG uses decentralized actors and centralized critics.

- Actor: one per UAV, `local_obs -> continuous_action`
- Critic: one per UAV, `(joint_obs, joint_actions) -> Q_i`
- Replay buffer: one shared joint replay buffer
- Exploration: OU noise added to each UAV action
- Target update: soft Polyak update with `tau`
- Cooperative reward: the mean team reward is broadcast to every critic
- Actions are projected onto the unit disk, so `max_step_size` is the actual
  maximum movement distance
- Replay warm-up, delayed actor updates, Huber critic loss, and reward scaling
  limit critic overestimation and actor saturation
- Collision-aware waypoint guidance seeds useful replay trajectories and adds
  a decaying behavior-cloning term; evaluation still uses actors alone
- Execution: decentralized actor inference with no centralized critic needed

Current defaults:

- episodes: `5000`
- replay buffer: `600000` joint transitions
- batch size: `256`
- hidden dim: `256`
- actor lr: `1e-4`
- critic lr: `3e-4`
- tau: `0.005`
- noise sigma: `0.35 -> 0.03`, decay `0.9992`
- reward scale: `0.01`
- replay warm-up: `5000` transitions
- learning interval: every `4` environment steps
- actor update interval: every `2` critic updates
- guidance mix: `0.5 -> 0` over `50000` environment steps
- behavior-cloning weight: `1.0 -> 0.05`
- max steps: `1000`

MADDQN checkpoints are not compatible with MADDPG because the action space and
network structure are different.

## Run

From the repository root:

```powershell
cd ".\Multi-Agent\2_5_MADDPG"
..\..\venv\Scripts\python.exe .\train.py
```

Regenerate plots from an existing checkpoint:

```powershell
cd ".\Multi-Agent\2_5_MADDPG"
..\..\venv\Scripts\python.exe .\test.py
```

## Results

Training writes these files under `results/`:

- `training_log.csv`
- `maddpg_model.pt` (best periodic greedy-evaluation checkpoint)
- `maddpg_final.pt` (last training checkpoint)
- `maddpg_best_success.pt`
- `training_curve.png`
- `success_curve.png`
- `critic_loss.png`
- `actor_loss.png`
- `best_path.png`
- `flight.gif`
