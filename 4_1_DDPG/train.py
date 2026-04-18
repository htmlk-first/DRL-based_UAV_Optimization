"""
DDPG 학습 및 평가 스크립트
─────────────────────────────────
DDQN 대비 변경:
  - DDPGAgent (Actor-Critic) 사용
  - 연속 행동 공간: ε-greedy 없이 OU Noise로 탐색
  - Critic + Actor 손실 각각 기록 및 시각화
  - Noise sigma 감쇠 (에피소드 단위)
"""
import os
import csv
import numpy as np

from env.config import EnvConfig
from env.uav_env import UAVEnv
from visualize import (plot_reward_curve, plot_path,
                       plot_critic_loss, plot_actor_loss,
                       plot_success_curve, make_flight_gif)
from ddpg_agent import DDPGAgent


def train(config, agent, n_episodes=2000, print_every=500, log_path=None):
    env = UAVEnv(config)
    rewards_history = []
    success_history = []
    critic_losses = []
    actor_losses = []

    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "reward", "success", "noise_sigma",
                              "ep_avg_critic_loss", "ep_avg_actor_loss", "buffer_size"])

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        agent.noise.reset()
        total_reward = 0
        done = False
        ep_critic_losses = []
        ep_actor_losses = []

        while not done:
            action = agent.select_action(obs, add_noise=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, float(done))
            c_loss, a_loss = agent.learn()
            if c_loss is not None:
                critic_losses.append(c_loss)
                ep_critic_losses.append(c_loss)
            if a_loss is not None:
                actor_losses.append(a_loss)
                ep_actor_losses.append(a_loss)

            obs = next_obs
            total_reward += reward

        agent.decay_noise()
        success = 1 if info["event"] == "mission_complete" else 0
        rewards_history.append(total_reward)
        success_history.append(success)

        if csv_writer is not None:
            ep_avg_cl = float(np.mean(ep_critic_losses)) if ep_critic_losses else float("nan")
            ep_avg_al = float(np.mean(ep_actor_losses)) if ep_actor_losses else float("nan")
            csv_writer.writerow([ep, total_reward, success, agent.noise.sigma,
                                  ep_avg_cl, ep_avg_al, len(agent.buffer)])

        if ep % print_every == 0:
            avg_r = np.mean(rewards_history[-print_every:])
            avg_s = np.mean(success_history[-print_every:]) * 100
            avg_cl = np.mean(critic_losses[-1000:]) if critic_losses else 0
            avg_al = np.mean(actor_losses[-1000:]) if actor_losses else 0
            print(f"[Ep {ep:5d}] AvgReward: {avg_r:8.1f} | "
                  f"Success: {avg_s:5.1f}% | "
                  f"Noise σ: {agent.noise.sigma:.4f} | "
                  f"CriticL: {avg_cl:.4f} | ActorL: {avg_al:.4f} | "
                  f"Buffer: {len(agent.buffer)}")

    if csv_file is not None:
        csv_file.close()

    return rewards_history, success_history, critic_losses, actor_losses


def evaluate(config, agent, n_episodes=10):
    env = UAVEnv(config)

    total_rewards = []
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs, add_noise=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)
        if info["event"] == "mission_complete":
            successes += 1

    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    print(f"Avg Reward: {np.mean(total_rewards):.1f}")
    print(f"Success Rate: {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")

    env.render()
    print(f"Path length: {len(env.path)}")
    return env


if __name__ == "__main__":
    # ── 설정 ──
    config = EnvConfig(
        grid_size=30,
        obstacle_mode="fixed",
        energy_budget_multiplier=3.0,
        max_step_size=1.5,
        wp_reach_radius=1.0,
    )

    tmp_env = UAVEnv(config)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]

    agent = DDPGAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=128,
        buffer_capacity=80000,
        hidden_dim=256,
        noise_sigma=0.3,
        noise_sigma_min=0.01,
        noise_decay=0.9972,
        grad_clip=1.0,
    )

    # ── 저장 경로 ──
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    # ── 학습 ──
    print("=== DDPG Training Start ===")
    print(f"Grid: {config.grid_size}x{config.grid_size}")
    print(f"Waypoints: {config.waypoints}")
    print(f"Energy budget: {config.compute_energy_budget():.0f}")
    print(f"Max step size: {config.max_step_size}")
    print(f"WP reach radius: {config.wp_reach_radius}")
    print(f"State dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Device: {agent.device}")

    rewards, successes, c_losses, a_losses = train(
        config, agent, n_episodes=2000, print_every=500,
        log_path=os.path.join(save_dir, "training_log.csv"),
    )

    # ── 모델 저장 ──
    agent.save(os.path.join(save_dir, "ddpg_model.pt"))

    # ── 학습 곡선 ──
    plot_reward_curve(rewards, window=50, title="DDPG Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))

    plot_success_curve(successes, window=50, title="DDPG Success Rate",
                       save_path=os.path.join(save_dir, "success_curve.png"))

    if c_losses:
        plot_critic_loss(c_losses, window=50, title="DDPG Critic Loss",
                         save_path=os.path.join(save_dir, "critic_loss.png"))

    if a_losses:
        plot_actor_loss(a_losses, window=50, title="DDPG Actor Loss",
                        save_path=os.path.join(save_dir, "actor_loss.png"))

    # ── 평가 ──
    env = evaluate(config, agent, n_episodes=10)

    # ── 경로 시각화 ──
    plot_path(env, title="DDPG - Best Path",
              save_path=os.path.join(save_dir, "best_path.png"))

    # ── GIF ──
    make_flight_gif(env, title="DDPG Flight",
                    save_path=os.path.join(save_dir, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {save_dir} ===")
