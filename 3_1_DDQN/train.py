import os
import sys
import csv
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.experiment import ExperimentPaths, print_eval_summary, success_from_info
from env.config import EnvConfig
from env.uav_env import UAVEnv
from visualize import (plot_reward_curve, plot_path,
                       plot_loss_curve, plot_success_curve,
                       make_flight_gif)
from ddqn_agent import DDQNAgent


def train(config, agent, n_episodes=4000, print_every=500, log_path=None):
    env = UAVEnv(config)
    rewards_history = []
    success_history = []
    loss_history = []

    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "reward", "success", "epsilon",
                              "ep_avg_loss", "buffer_size"])

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        ep_losses = []

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, float(done))
            loss = agent.learn()
            if loss is not None:
                loss_history.append(loss)
                ep_losses.append(loss)

            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()
        success = success_from_info(info)
        rewards_history.append(total_reward)
        success_history.append(success)

        if csv_writer is not None:
            ep_avg_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            csv_writer.writerow([ep, total_reward, success, agent.epsilon,
                                  ep_avg_loss, len(agent.buffer)])

        if ep % print_every == 0:
            avg_r = np.mean(rewards_history[-print_every:])
            avg_s = np.mean(success_history[-print_every:]) * 100
            avg_l = np.mean(loss_history[-1000:]) if loss_history else 0
            print(f"[Ep {ep:5d}] AvgReward: {avg_r:8.1f} | "
                  f"Success: {avg_s:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"AvgLoss: {avg_l:.4f} | "
                  f"Buffer: {len(agent.buffer)}")

    if csv_file is not None:
        csv_file.close()

    return rewards_history, success_history, loss_history


def evaluate(config, agent, n_episodes=10):
    env = UAVEnv(config)
    original_eps = agent.epsilon
    agent.epsilon = 0.0

    total_rewards = []
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)
        successes += success_from_info(info)

    agent.epsilon = original_eps

    print_eval_summary(total_rewards, successes, n_episodes)

    env.render()
    print(f"Path length: {len(env.path)}")
    return env


if __name__ == "__main__":
    # ── 설정 ──
    config = EnvConfig(
        grid_size=50,
        obstacle_mode="fixed",
        energy_budget_multiplier=2.8,
        max_steps=1500,
    )

    tmp_env = UAVEnv(config)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n

    agent = DDQNAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.9985,
        batch_size=128,
        buffer_capacity=250000,
        target_update_freq=1000,
        hidden_dim=256,
        grad_clip=5.0,
    )

    # ── 저장 경로 ──
    paths = ExperimentPaths.from_file(__file__)
    save_dir = str(paths.results_dir)

    # ── 학습 ──
    print("=== Double DQN Training Start ===")
    print(f"Grid: {config.grid_size}x{config.grid_size}")
    print(f"Waypoints: {config.waypoints}")
    print(f"Obstacles: {len(config.get_obstacles())}")
    print(f"Energy budget: {config.compute_energy_budget():.0f}")
    print(f"Max steps: {config.max_steps}")
    print(f"State dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Device: {agent.device}")
    rewards, successes, losses = train(
        config, agent, n_episodes=4000, print_every=500,
        log_path=os.path.join(save_dir, "training_log.csv"),
    )

    # ── 모델 저장 ──
    agent.save(os.path.join(save_dir, "ddqn_model.pt"))

    # ── 학습 곡선 ──
    plot_reward_curve(rewards, window=50, title="DDQN Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))

    plot_success_curve(successes, window=50, title="DDQN Success Rate",
                       save_path=os.path.join(save_dir, "success_curve.png"))

    if losses:
        plot_loss_curve(losses, window=50, title="DDQN Training Loss",
                        save_path=os.path.join(save_dir, "loss_curve.png"))

    # ── 평가 ──
    env = evaluate(config, agent, n_episodes=10)

    # ── 경로 시각화 ──
    plot_path(env, title="DDQN - Best Path",
              save_path=os.path.join(save_dir, "best_path.png"))

    # ── GIF ──
    make_flight_gif(env, title="DDQN Flight",
                    save_path=os.path.join(save_dir, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {save_dir} ===")
