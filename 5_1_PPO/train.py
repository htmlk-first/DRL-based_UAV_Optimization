"""
PPO 학습 및 평가 스크립트
─────────────────────────────────
DDPG 대비 변경:
  - PPOAgent (Actor-Critic, 단일 네트워크) 사용
  - On-Policy: 에피소드 단위 rollout → GAE → K epochs 학습
  - OU Noise 제거 → 정책 분포 자체의 확률적 탐색 + 엔트로피 보너스
  - Policy Loss / Value Loss / Entropy 기록 및 시각화
  - 학습률 스케줄링 (LR Annealing)
"""
import os
import numpy as np

from env.config import EnvConfig
from env.uav_env import UAVEnv
from visualize import (plot_reward_curve, plot_path,
                       plot_policy_loss, plot_value_loss,
                       plot_entropy_curve,
                       plot_success_curve, make_flight_gif)
from ppo_agent import PPOAgent


def train(config, agent, n_episodes=4000, print_every=500):
    env = UAVEnv(config)
    rewards_history = []
    success_history = []
    policy_losses = []
    value_losses = []
    entropies = []

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
        # last_value: 에피소드가 끝나지 않았을 때의 부트스트랩 값
        if not terminated:
            _, _, last_value = agent.select_action(obs, deterministic=True)
        else:
            last_value = 0.0

        p_loss, v_loss, entropy = agent.learn(last_value)
        if p_loss is not None:
            policy_losses.append(p_loss)
            value_losses.append(v_loss)
            entropies.append(entropy)

        rewards_history.append(total_reward)
        success_history.append(1 if info["event"] == "mission_complete" else 0)

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

    return rewards_history, success_history, policy_losses, value_losses, entropies


def evaluate(config, agent, n_episodes=10):
    env = UAVEnv(config)

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
                env = UAVEnv(config)

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
                    env = UAVEnv(config)

    result_env = best_env if best_env is not None else env

    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    print(f"Avg Reward: {np.mean(total_rewards):.1f}")
    print(f"Success Rate: {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")

    result_env.render()
    print(f"Path length: {len(result_env.path)}")
    return result_env


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

    N_EPISODES = 6000

    agent = PPOAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.001,
        value_coeff=0.5,
        max_grad_norm=0.5,
        k_epochs=10,
        mini_batch_size=64,
        hidden_dim=256,
        lr_decay=True,
        total_updates=N_EPISODES,
    )

    # ── 학습 ──
    print("=== PPO Training Start ===")
    print(f"Grid: {config.grid_size}x{config.grid_size}")
    print(f"Waypoints: {config.waypoints}")
    print(f"Energy budget: {config.compute_energy_budget():.0f}")
    print(f"Max step size: {config.max_step_size}")
    print(f"WP reach radius: {config.wp_reach_radius}")
    print(f"State dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Device: {agent.device}")
    print(f"LR: {agent.initial_lr}, K epochs: {agent.k_epochs}, "
          f"Mini-batch: {agent.mini_batch_size}")
    print(f"GAE λ: {agent.gae_lambda}, Clip ε: {agent.clip_epsilon}, "
          f"Entropy coeff: {agent.entropy_coeff}")

    rewards, successes, p_losses, v_losses, ents = train(
        config, agent, n_episodes=N_EPISODES, print_every=500
    )

    # ── 저장 ──
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, "ppo_model.pt"))

    # ── 학습 곡선 ──
    plot_reward_curve(rewards, window=50, title="PPO Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))

    plot_success_curve(successes, window=50, title="PPO Success Rate",
                       save_path=os.path.join(save_dir, "success_curve.png"))

    if p_losses:
        plot_policy_loss(p_losses, window=50, title="PPO Policy Loss",
                         save_path=os.path.join(save_dir, "policy_loss.png"))

    if v_losses:
        plot_value_loss(v_losses, window=50, title="PPO Value Loss",
                        save_path=os.path.join(save_dir, "value_loss.png"))

    if ents:
        plot_entropy_curve(ents, window=50, title="PPO Entropy",
                           save_path=os.path.join(save_dir, "entropy.png"))

    # ── 평가 ──
    env = evaluate(config, agent, n_episodes=10)

    # ── 경로 시각화 ──
    plot_path(env, title="PPO - Best Path",
              save_path=os.path.join(save_dir, "best_path.png"))

    # ── GIF ──
    make_flight_gif(env, title="PPO Flight",
                    save_path=os.path.join(save_dir, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {save_dir} ===")
