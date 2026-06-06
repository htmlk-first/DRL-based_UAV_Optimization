# 2_3_QMIX_MADQN: CTDE QMIX-MADDQN

This experiment keeps the 2D cooperative environment and script layout from
`2_1_MADQN`, then replaces independent MADQN training with QMIX-style
centralized training. Execution remains decentralized: each UAV selects its
own action from the shared per-agent Q-network.

## CTDE Structure

QMIX learns a monotonic mixing function:

```text
Q_total = Mixer(Q_0(o_0, a_0), Q_1(o_1, a_1), Q_2(o_2, a_2), Q_3(o_3, a_3), s)
```

- Centralized training: one replay item stores the full joint transition
  `(obs_batch, joint_actions, team_reward, next_obs_batch, done)`.
- Mixer input: the flattened joint observation is used as the global state.
- QMIX loss: Bellman target is computed on `Q_total`.
- Double DQN target: the online Q-network selects next actions, while the
  target Q-network and target mixer evaluate them.
- Decentralized execution: each UAV selects `argmax_a Q_i(o_i, a)` using only
  its own observation row.
- Invalid wall, obstacle, and occupied-cell actions are masked during both
  exploration and Double-DQN target selection.
- Each local observation includes the previous action and a failed-action
  flag, allowing a deterministic policy to break repeated collision loops.
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
- replay warm-up: `10000` transitions
- learning interval: every `4` environment steps
- soft target update: `tau=0.005`
- hidden dim: `256`
- QMIX mixing embed dim: `32`
- learning reward scale: `0.01`
- max steps: `1000`

Old checkpoints from MADQN, MADDQN, or VDN-MADQN are not compatible with this
agent because QMIX adds mixer parameters.

## Run

From the repository root:

```powershell
cd ".\Multi-Agent\2_3_QMIX_MADQN"
..\..\venv\Scripts\python.exe .\train.py
```

Regenerate plots from an existing checkpoint:

```powershell
cd ".\Multi-Agent\2_3_QMIX_MADQN"
..\..\venv\Scripts\python.exe .\test.py
```

## Results

Training writes these files under `results/`:

- `training_log.csv`
- `qmix_maddqn_model.pt` (best periodic greedy-evaluation checkpoint)
- `qmix_maddqn_final.pt` (last training checkpoint)
- `qmix_maddqn_best_success.pt` (best periodic greedy evaluation)
- `training_curve.png`
- `success_curve.png`
- `loss_curve.png`
- `best_path.png`
- `flight.gif`
