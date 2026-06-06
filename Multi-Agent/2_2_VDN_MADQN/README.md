# 2_2_VDN_MADQN: CTDE Value-Decomposition MADQN

This experiment is the CTDE version of `2_1_MADQN`.
It uses the same cooperative four-UAV environment, but trains with a VDN
joint value target instead of independent per-agent DQN transitions.

## CTDE Structure

VDN decomposes the team value as:

```text
Q_total = Q_0(o_0, a_0) + Q_1(o_1, a_1) + Q_2(o_2, a_2) + Q_3(o_3, a_3)
```

- Centralized training: one replay item stores the full joint transition
  `(obs_batch, joint_actions, team_reward, next_obs_batch, done)`.
- VDN loss: Bellman target is computed on `Q_total`.
- Decentralized execution: each UAV selects `argmax_a Q_i(o_i, a)` using only
  its own observation row.
- Parameter sharing: all UAVs use the same per-agent Q-network; the one-hot
  agent id in the observation lets the shared network distinguish roles.

## Environment

- 2D grid: `20x20`
- Agents: 4 UAVs, default starts at the four map corners
- Action space: `MultiDiscrete([4, 4, 4, 4])`
- Mission: all UAVs share ten unordered waypoints and complete coverage together
- Waypoint reward: only the first UAV visit to each waypoint receives the waypoint bonus
- Collision rules: wall, obstacle, same-cell entry, and swap collisions block movement

Current harder-task defaults:

- episodes: `10000`
- epsilon: `1.0 -> 0.05`, decay `0.9992`
- replay buffer: `600000` joint transitions
- batch size: `256`
- target update: `1500` learning steps
- hidden dim: `512`
- max steps: `1000`

Old 2-UAV checkpoints must be retrained because the 4-UAV / 10-waypoint
observation dimension is different.

## Run

From the repository root:

```powershell
cd ".\Multi-Agent\2_2_VDN_MADQN"
..\..\venv\Scripts\python.exe .\train.py
```

Regenerate plots from an existing checkpoint:

```powershell
cd ".\Multi-Agent\2_2_VDN_MADQN"
..\..\venv\Scripts\python.exe .\test.py
```

## Results

Training writes these files under `results/`:

- `training_log.csv`
- `vdn_madqn_model.pt`
- `training_curve.png`
- `success_curve.png`
- `loss_curve.png`
- `best_path.png`
- `flight.gif`
