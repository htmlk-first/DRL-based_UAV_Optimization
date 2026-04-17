"""
PPO 3D 학습 및 평가 스크립트
─────────────────────────────────
5_1_PPO 대비 변경:
  - EnvConfig3D, UAVEnv3D 사용 (30×30×5 3D 공간)
  - 3D 행동 공간 (dx, dy, dz)
  - 3D 시각화 (plot_path_3d, make_flight_gif_3d)
  - z축 추가 에너지 비용 고려
  - PPO 핵심 알고리즘은 동일 (On-Policy, GAE, Clipped Surrogate)
"""
import os
import csv
import numpy as np

from env.config import EnvConfig3D
from env.uav_env import UAVEnv3D
from visualize import (plot_reward_curve, plot_path_3d,
                       plot_policy_loss, plot_value_loss,
                       plot_entropy_curve,
                       plot_success_curve, make_flight_gif_3d)
from ppo_agent import PPOAgent


def train(config, agent, n_episodes=4000, print_every=500, log_path=None):
    env = UAVEnv3D(config)
    rewards_history = []
    success_history = []
    policy_losses = []
    value_losses = []
    entropies = []

    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "reward", "success",
                              "policy_loss", "value_loss", "entropy", "learning_rate"])

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        # ── 에피소드 rollout 수집 ──
        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, log_prob, reward, value, float(done))
            obs = next_obs
            total_reward += reward

        # ── 에피소드 종료 후 PPO 학습 ──
        if not terminated:
            _, _, last_value = agent.select_action(obs, deterministic=True)
        else:
            last_value = 0.0

        p_loss, v_loss, entropy = agent.learn(last_value)
        if p_loss is not None:
            policy_losses.append(p_loss)
            value_losses.append(v_loss)
            entropies.append(entropy)

        success = 1 if info["event"] == "mission_complete" else 0
        rewards_history.append(total_reward)
        success_history.append(success)

        if csv_writer is not None:
            p_loss_val = float(p_loss) if p_loss is not None else float("nan")
            v_loss_val = float(v_loss) if v_loss is not None else float("nan")
            entropy_val = float(entropy) if entropy is not None else float("nan")
            csv_writer.writerow([ep, total_reward, success,
                                  p_loss_val, v_loss_val, entropy_val, agent.current_lr()])

        if ep % print_every == 0:
            avg_r = np.mean(rewards_history[-print_every:])
            avg_s = np.mean(success_history[-print_every:]) * 100
            avg_pl = np.mean(policy_losses[-200:]) if policy_losses else 0
            avg_vl = np.mean(value_losses[-200:]) if value_losses else 0
            avg_ent = np.mean(entropies[-200:]) if entropies else 0
            lr = agent.current_lr()
            print(f"[Ep {ep:5d}] AvgReward: {avg_r:8.1f} | "
                  f"Success: {avg_s:5.1f}% | "
                  f"PolicyL: {avg_pl:.4f} | ValueL: {avg_vl:.4f} | "
                  f"Entropy: {avg_ent:.4f} | LR: {lr:.2e}")

    if csv_file is not None:
        csv_file.close()

    return rewards_history, success_history, policy_losses, value_losses, entropies


def evaluate(config, agent, n_episodes=10):
    env = UAVEnv3D(config)

    total_rewards = []
    successes = 0
    best_env = None
    best_reward = -float('inf')

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)
        if info["event"] == "mission_complete":
            successes += 1
            if total_reward > best_reward:
                best_reward = total_reward
                best_env = env
                env = UAVEnv3D(config)

    # deterministic 실패 시 stochastic으로 재평가
    if successes == 0:
        print("\n[Deterministic eval failed, retrying with stochastic policy...]")
        total_rewards = []
        successes = 0
        best_env = None
        best_reward = -float('inf')

        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action, _, _ = agent.select_action(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

            total_rewards.append(total_reward)
            if info["event"] == "mission_complete":
                successes += 1
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_env = env
                    env = UAVEnv3D(config)

    result_env = best_env if best_env is not None else env

    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    print(f"Avg Reward: {np.mean(total_rewards):.1f}")
    print(f"Success Rate: {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")

    result_env.render()
    print(f"Path length: {len(result_env.path)}")
    return result_env


if __name__ == "__main__":
    # ── 설정 ──
    config = EnvConfig3D(
        grid_size_x=30,
        grid_size_y=30,
        grid_size_z=5,
        obstacle_mode="fixed",
        num_buildings=40,
        energy_budget_multiplier=4.0,
        max_step_size=1.5,
        max_step_size_z=1.0,
        wp_reach_radius=1.0,
        z_cost_multiplier=1.5,
    )

    tmp_env = UAVEnv3D(config)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]

    N_EPISODES = 10000

    agent = PPOAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        max_grad_norm=0.5,
        k_epochs=10,
        mini_batch_size=64,
        hidden_dim=256,
        lr_decay=True,
        total_updates=N_EPISODES,
    )

    # ── 저장 경로 ──
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    # ── 학습 ──
    print("=== PPO 3D Training Start ===")
    print(f"Grid: {config.grid_size_x}x{config.grid_size_y}x{config.grid_size_z}")
    print(f"Waypoints: {config.waypoints}")
    print(f"Energy budget: {config.compute_energy_budget():.0f}")
    print(f"Max step size (XY): {config.max_step_size}, (Z): {config.max_step_size_z}")
    print(f"WP reach radius: {config.wp_reach_radius}")
    print(f"Z cost multiplier: {config.z_cost_multiplier}")
    print(f"State dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Device: {agent.device}")
    print(f"LR: {agent.initial_lr}, K epochs: {agent.k_epochs}, "
          f"Mini-batch: {agent.mini_batch_size}")
    print(f"GAE λ: {agent.gae_lambda}, Clip ε: {agent.clip_epsilon}, "
          f"Entropy coeff: {agent.entropy_coeff}")

    rewards, successes, p_losses, v_losses, ents = train(
        config, agent, n_episodes=N_EPISODES, print_every=500,
        log_path=os.path.join(save_dir, "training_log.csv"),
    )

    # ── 모델 저장 ──
    agent.save(os.path.join(save_dir, "ppo_3d_model.pt"))

    # ── 학습 곡선 ──
    plot_reward_curve(rewards, window=50, title="PPO 3D Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))

    plot_success_curve(successes, window=50, title="PPO 3D Success Rate",
                       save_path=os.path.join(save_dir, "success_curve.png"))

    if p_losses:
        plot_policy_loss(p_losses, window=50, title="PPO 3D Policy Loss",
                         save_path=os.path.join(save_dir, "policy_loss.png"))

    if v_losses:
        plot_value_loss(v_losses, window=50, title="PPO 3D Value Loss",
                        save_path=os.path.join(save_dir, "value_loss.png"))

    if ents:
        plot_entropy_curve(ents, window=50, title="PPO 3D Entropy",
                           save_path=os.path.join(save_dir, "entropy.png"))

    # ── 평가 ──
    env = evaluate(config, agent, n_episodes=10)

    # ── 경로 시각화 ──
    plot_path_3d(env, title="PPO 3D - Best Path",
                 save_path=os.path.join(save_dir, "best_path_3d.png"))

    # ── GIF ──
    make_flight_gif_3d(env, title="PPO 3D Flight",
                       save_path=os.path.join(save_dir, "flight_3d.gif"), fps=5)

    print(f"\n=== Results saved to {save_dir} ===")
