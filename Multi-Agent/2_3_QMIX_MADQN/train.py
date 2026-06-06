"""Train CTDE QMIX-MADDQN on the cooperative 2D UAV waypoint task."""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common.experiment import ExperimentPaths, print_eval_summary, success_from_info
from env.config import MultiAgentEnvConfig
from env.multi_uav_env import MultiUAVEnv
from qmix_maddqn_agent import QMIXMADDQNAgent
from visualize import (
    make_flight_gif,
    plot_loss_curve,
    plot_path,
    plot_reward_curve,
    plot_success_curve,
)


LEARNING_REWARD_SCALE = 0.01


def scalar_team_reward(rewards, scale=1.0):
    """Team scalar used by QMIX; includes shared rewards and local penalties."""
    return float(np.mean(rewards)) * scale


def rollout_episode(config, agent):
    """Run one episode using the agent's current epsilon."""
    env = MultiUAVEnv(config)
    obs, _ = env.reset()
    action_masks = env.get_action_masks()
    done = False
    episode_return = 0.0
    info = {}

    while not done:
        actions = agent.select_actions(obs, action_masks)
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        action_masks = env.get_action_masks()
        episode_return += scalar_team_reward(rewards)

    return episode_return, success_from_info(info), info, env


def greedy_rollout(config, agent):
    """Evaluate once without exploration and restore the training epsilon."""
    original_eps = agent.epsilon
    agent.epsilon = 0.0
    try:
        return rollout_episode(config, agent)
    finally:
        agent.epsilon = original_eps


def train(config, agent, n_episodes=10000, print_every=500, log_path=None):
    env = MultiUAVEnv(config)
    rewards_history = []
    success_history = []
    loss_history = []
    collision_history = []

    csv_file = None
    csv_writer = None
    best_eval_score = (-1, -1, -float("inf"))
    best_eval_return = -float("inf")
    best_success_path = None
    best_return_path = None
    if log_path is not None:
        results_dir = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(results_dir, exist_ok=True)
        best_success_path = os.path.join(results_dir, "qmix_maddqn_best_success.pt")
        best_return_path = os.path.join(results_dir, "qmix_maddqn_best_return.pt")
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "episode",
            "reward",
            "avg_team_return",
            "success",
            "epsilon",
            "ep_avg_loss",
            "buffer_size",
            "steps",
            "collisions",
            "waypoints_visited",
            "greedy_reward",
            "greedy_success",
            "greedy_waypoints_visited",
        ])

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        action_masks = env.get_action_masks()
        done = False
        ep_team_return = 0.0
        ep_losses = []
        info = {}

        while not done:
            actions = agent.select_actions(obs, action_masks)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            next_action_masks = env.get_action_masks()
            team_reward = scalar_team_reward(rewards)
            learn_reward = scalar_team_reward(rewards, scale=LEARNING_REWARD_SCALE)

            agent.store_joint(
                obs,
                actions,
                learn_reward,
                next_obs,
                done,
                next_action_masks,
            )
            loss = agent.learn()
            if loss is not None:
                loss_history.append(loss)
                ep_losses.append(loss)

            obs = next_obs
            action_masks = next_action_masks
            ep_team_return += team_reward

        agent.decay_epsilon()
        success = success_from_info(info)
        rewards_history.append(ep_team_return)
        success_history.append(success)
        collision_history.append(info.get("collisions", 0))

        greedy_reward = float("nan")
        greedy_success = ""
        greedy_waypoints = ""
        should_evaluate = ep % print_every == 0 or ep == n_episodes
        if should_evaluate:
            (
                greedy_reward,
                greedy_success,
                greedy_info,
                _,
            ) = greedy_rollout(config, agent)
            greedy_waypoints = greedy_info.get("waypoints_visited", 0)
            eval_score = (
                int(greedy_success),
                int(greedy_waypoints),
                float(greedy_reward),
            )
            if best_success_path is not None and eval_score > best_eval_score:
                best_eval_score = eval_score
                agent.save(best_success_path)
            if (
                best_return_path is not None
                and greedy_reward > best_eval_return
            ):
                best_eval_return = greedy_reward
                agent.save(best_return_path)

        if csv_writer is not None:
            ep_avg_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            csv_writer.writerow([
                ep,
                ep_team_return,
                ep_team_return,
                success,
                agent.epsilon,
                ep_avg_loss,
                len(agent.buffer),
                info.get("steps", 0),
                info.get("collisions", 0),
                info.get("waypoints_visited", 0),
                greedy_reward,
                greedy_success,
                greedy_waypoints,
            ])
            csv_file.flush()

        if should_evaluate:
            avg_r = float(np.mean(rewards_history[-print_every:]))
            avg_s = float(np.mean(success_history[-print_every:]) * 100)
            avg_l = float(np.mean(loss_history[-1000:])) if loss_history else 0.0
            avg_c = float(np.mean(collision_history[-print_every:]))
            print(f"[Ep {ep:5d}] AvgTeamReturn: {avg_r:8.1f} | "
                  f"Success: {avg_s:5.1f}% | "
                  f"Greedy: {int(greedy_success)}/"
                  f"{int(greedy_waypoints)} WP, {greedy_reward:7.1f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"AvgQMIXLoss: {avg_l:.4f} | "
                  f"AvgCollisions: {avg_c:.1f} | "
                  f"JointBuffer: {len(agent.buffer)}")

    if csv_file is not None:
        csv_file.close()

    return rewards_history, success_history, loss_history


def evaluate(config, agent, n_episodes=10):
    original_eps = agent.epsilon
    agent.epsilon = 0.0

    best_reward = -float("inf")
    best_env = None
    total_rewards = []
    successes = 0

    for _ in range(n_episodes):
        ep_reward, success, info, env = rollout_episode(config, agent)

        total_rewards.append(ep_reward)
        successes += success
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_env = env

    agent.epsilon = original_eps
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
    obs_dim = tmp_env.single_observation_dim
    act_dim = 4

    agent = QMIXMADDQNAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
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

    paths = ExperimentPaths.from_file(__file__)
    save_dir = str(paths.results_dir)

    print("=== QMIX-MADDQN Training Start ===")
    print(f"Agents: {config.n_agents}, State dim: {obs_dim}, Action dim: {act_dim}")
    print("Training: centralized QMIX loss, Execution: decentralized per-agent argmax")
    print("Target: Double DQN")
    print("Action masking: wall, obstacle, and occupied-cell actions")
    print(f"Learning reward scale: {LEARNING_REWARD_SCALE}")
    print(
        f"Warm-up: {agent.warmup_steps} transitions | "
        f"Learn every: {agent.learn_every} steps | "
        f"Target tau: {agent.target_tau}"
    )
    print(f"Device: {agent.device}")

    rewards, successes, losses = train(
        config,
        agent,
        n_episodes=10000,
        print_every=500,
        log_path=os.path.join(save_dir, "training_log.csv"),
    )

    final_model_path = os.path.join(save_dir, "qmix_maddqn_final.pt")
    best_model_path = os.path.join(save_dir, "qmix_maddqn_best_success.pt")
    canonical_model_path = os.path.join(save_dir, "qmix_maddqn_model.pt")
    agent.save(final_model_path)
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
    agent.save(canonical_model_path)

    plot_reward_curve(rewards, window=100, title="QMIX-MADDQN Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))
    plot_success_curve(successes, window=100, title="QMIX-MADDQN Success Rate",
                       save_path=os.path.join(save_dir, "success_curve.png"))
    if losses:
        plot_loss_curve(losses, window=200, title="QMIX-MADDQN Training Loss",
                        save_path=os.path.join(save_dir, "loss_curve.png"))

    best_env = evaluate(config, agent, n_episodes=10)
    if best_env is not None:
        plot_path(best_env, title="QMIX-MADDQN - Best Cooperative Path",
                  save_path=os.path.join(save_dir, "best_path.png"))
        make_flight_gif(best_env, title="QMIX-MADDQN Cooperative Flight",
                        save_path=os.path.join(save_dir, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {save_dir} ===")
