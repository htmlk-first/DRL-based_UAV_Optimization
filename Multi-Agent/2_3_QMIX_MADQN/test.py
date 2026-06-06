"""QMIX-MADDQN Test: loads saved model and regenerates result plots."""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common.experiment import (
    ExperimentPaths,
    optional_float,
    print_eval_summary,
    read_training_log,
    success_from_info,
)
from env.config import MultiAgentEnvConfig
from env.multi_uav_env import MultiUAVEnv
from train import scalar_team_reward
from qmix_maddqn_agent import QMIXMADDQNAgent
from visualize import (
    make_flight_gif,
    plot_loss_curve,
    plot_path,
    plot_reward_curve,
    plot_success_curve,
)

RESULTS = str(ExperimentPaths.from_file(__file__).results_dir)


def load_log():
    rewards, successes, losses = [], [], []
    for row in read_training_log(RESULTS):
        rewards.append(float(row["reward"]))
        successes.append(int(row["success"]))
        loss = optional_float(row, "ep_avg_loss")
        if loss is not None:
            losses.append(loss)
    return rewards, successes, losses


def evaluate(config, agent, n_episodes=10):
    agent.epsilon = 0.0
    best_reward = -float("inf")
    best_env = None
    successes = 0
    total_rewards = []

    for _ in range(n_episodes):
        env = MultiUAVEnv(config)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info = {}

        while not done:
            action_masks = env.get_action_masks()
            actions = agent.select_actions(obs, action_masks)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            ep_reward += scalar_team_reward(rewards)

        total_rewards.append(ep_reward)
        successes += success_from_info(info)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_env = env

    print_eval_summary(total_rewards, successes, n_episodes,
                       reward_label="Avg Team Return", success_label="Success Rate")
    if best_env is not None:
        best_env.render()
        print(f"Path lengths: {[len(path) for path in best_env.paths]}")
    return best_env


if __name__ == "__main__":
    config = MultiAgentEnvConfig(
        n_agents=4,
        start_positions=[(0, 0), (0, 19), (19, 0), (19, 19)],
        waypoints=[
            (4, 4),
            (4, 10),
            (4, 15),
            (8, 6),
            (8, 13),
            (12, 4),
            (12, 10),
            (12, 16),
            (16, 7),
            (16, 14),
        ],
        grid_size=20,
        obstacle_mode="fixed",
        ordered_waypoints=False,
        energy_budget_multiplier=2.2,
        collision_penalty=-20.0,
        max_steps=1000,
    )

    tmp_env = MultiUAVEnv(config)
    agent = QMIXMADDQNAgent(
        state_dim=tmp_env.single_observation_dim,
        action_dim=4,
        n_agents=config.n_agents,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9992,
        batch_size=256,
        buffer_capacity=600000,
        target_update_freq=3000,
        hidden_dim=256,
        mixing_embed_dim=32,
        grad_clip=1.0,
        warmup_steps=10000,
        learn_every=4,
        target_tau=0.005,
    )

    model_path = os.path.join(RESULTS, "qmix_maddqn_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run train.py before test.py."
        )
    try:
        agent.load(model_path)
    except RuntimeError as exc:
        raise RuntimeError(
            "The saved qmix_maddqn_model.pt is incompatible with the current "
            "4-UAV / 10-waypoint tuned QMIX architecture. Run train.py again "
            "to create a fresh checkpoint with the updated parameters."
        ) from exc
    agent.epsilon = 0.0

    rewards, successes, losses = load_log()
    plot_reward_curve(rewards, window=100, title="QMIX-MADDQN Training Curve",
                      save_path=os.path.join(RESULTS, "training_curve.png"))
    plot_success_curve(successes, window=100, title="QMIX-MADDQN Success Rate",
                       save_path=os.path.join(RESULTS, "success_curve.png"))
    if losses:
        plot_loss_curve(losses, window=200, title="QMIX-MADDQN Training Loss",
                        save_path=os.path.join(RESULTS, "loss_curve.png"))

    best_env = evaluate(config, agent, n_episodes=10)
    if best_env is not None:
        plot_path(best_env, title="QMIX-MADDQN - Best Cooperative Path",
                  save_path=os.path.join(RESULTS, "best_path.png"))
        make_flight_gif(best_env, title="QMIX-MADDQN Cooperative Flight",
                        save_path=os.path.join(RESULTS, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {RESULTS} ===")
