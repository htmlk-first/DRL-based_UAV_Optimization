"""
TQL 학습 및 평가 스크립트
"""
import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.experiment import ExperimentPaths, print_eval_summary, success_from_info
from env.config import EnvConfig
from env.uav_env import UAVEnv
from visualize import (plot_reward_curve, plot_path,
                       plot_qvalue_heatmap)
from tql_agent import TQLAgent


def train(config, agent, n_episodes=5000, print_every=500):
    env = UAVEnv(config)
    rewards_history = []
    success_history = []

    for ep in range(1, n_episodes + 1):
        env.reset()
        state = env.get_state_tuple()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = env.get_state_tuple()

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        success_history.append(success_from_info(info))

        if ep % print_every == 0:
            avg_r = np.mean(rewards_history[-print_every:])
            avg_s = np.mean(success_history[-print_every:]) * 100
            print(f"[Ep {ep:5d}] AvgReward: {avg_r:8.1f} | "
                  f"Success: {avg_s:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Q-table size: {len(agent.q_table)}")

    return rewards_history, success_history


def evaluate(config, agent, n_episodes=10, render_last=True):
    env = UAVEnv(config)
    agent.epsilon = 0.0  # greedy 평가

    total_rewards = []
    successes = 0

    for ep in range(n_episodes):
        env.reset()
        state = env.get_state_tuple()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = env.get_state_tuple()
            total_reward += reward

        total_rewards.append(total_reward)
        successes += success_from_info(info)

    print_eval_summary(total_rewards, successes, n_episodes)

    if render_last:
        env.render()
        print(f"Path length: {len(env.path)}")

    return env


if __name__ == "__main__":
    # ── 설정 ──
    config = EnvConfig(
        grid_size=10,
        obstacle_mode="fixed",
        energy_budget_multiplier=1.5
    )

    agent = TQLAgent(
        n_actions=4,
        lr=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
    )

    # ── 학습 ──
    print("=== TQL Training Start ===")
    rewards, successes = train(config, agent, n_episodes=5000, print_every=500)

    # ── 저장 ──
    paths = ExperimentPaths.from_file(__file__)
    save_dir = str(paths.results_dir)
    agent.save(os.path.join(save_dir, "tql_model.pkl"))

    # ── 학습 곡선 ──
    plot_reward_curve(rewards, window=100, title="TQL Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))

    # ── 평가 ──
    env = evaluate(config, agent, n_episodes=10, render_last=True)

    # ── Q-value 히트맵 ──
    tmp_env = UAVEnv(config)
    tmp_env.reset()
    for wp_i in range(tmp_env.n_waypoints):
        plot_qvalue_heatmap(
            agent, tmp_env,
            waypoint_idx=wp_i,
            title="TQL Q-Value Heatmap",
            save_path=os.path.join(save_dir, f"qvalue_heatmap_wp{wp_i}.png"),
        )

    # ── 경로 시각화 ──
    plot_path(env, title="TQL - Best Path",
              save_path=os.path.join(save_dir, "best_path.png"))
