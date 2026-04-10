"""
DQN 학습 및 평가 스크립트
"""
import os
import numpy as np

from env.config import EnvConfig
from env.uav_env import UAVEnv
from visualize import (plot_reward_curve, plot_path,
                       plot_loss_curve, plot_success_curve,
                       make_flight_gif)
from dqn_agent import DQNAgent


def train(config, agent, n_episodes=3000, print_every=500):
    env = UAVEnv(config)
    rewards_history = []
    success_history = []
    loss_history = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, float(done))
            loss = agent.learn()
            if loss is not None:
                loss_history.append(loss)

            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        success_history.append(1 if info["event"] == "mission_complete" else 0)

        if ep % print_every == 0:
            avg_r = np.mean(rewards_history[-print_every:])
            avg_s = np.mean(success_history[-print_every:]) * 100
            avg_l = np.mean(loss_history[-1000:]) if loss_history else 0
            print(f"[Ep {ep:5d}] AvgReward: {avg_r:8.1f} | "
                  f"Success: {avg_s:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"AvgLoss: {avg_l:.4f} | "
                  f"Buffer: {len(agent.buffer)}")

    return rewards_history, success_history, loss_history


def evaluate(config, agent, n_episodes=10):
    env = UAVEnv(config)
    original_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy 평가

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
        if info["event"] == "mission_complete":
            successes += 1

    agent.epsilon = original_eps

    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    print(f"Avg Reward: {np.mean(total_rewards):.1f}")
    print(f"Success Rate: {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")

    # 마지막 에피소드의 env 반환 (경로 시각화용)
    env.render()
    print(f"Path length: {len(env.path)}")
    return env


if __name__ == "__main__":
    # ── 설정 ──
    config = EnvConfig(
        grid_size=20,
        obstacle_mode="fixed",
        energy_budget_multiplier=2.0,
    )

    tmp_env = UAVEnv(config)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n

    agent = DQNAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        batch_size=64,
        buffer_capacity=100000,
        target_update_freq=500,
        hidden_dim=256,
    )

    # ── 학습 ──
    print("=== DQN Training Start ===")
    print(f"State dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Device: {agent.device}")
    rewards, successes, losses = train(config, agent, n_episodes=5000, print_every=500)

    # ── 저장 ──
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, "dqn_model.pt"))

    # ── 학습 곡선 ──
    plot_reward_curve(rewards, window=100, title="DQN Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))

    plot_success_curve(successes, window=100, title="DQN Success Rate",
                       save_path=os.path.join(save_dir, "success_curve.png"))

    if losses:
        plot_loss_curve(losses, window=200, title="DQN Training Loss",
                        save_path=os.path.join(save_dir, "loss_curve.png"))

    # ── 평가 ──
    env = evaluate(config, agent, n_episodes=10)

    # ── 경로 시각화 ──
    plot_path(env, title="DQN - Best Path",
              save_path=os.path.join(save_dir, "best_path.png"))

    # ── GIF ──
    make_flight_gif(env, title="DQN Flight",
                    save_path=os.path.join(save_dir, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {save_dir} ===")
