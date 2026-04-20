"""DDQN Test: loads saved model, regenerates result PNGs (no training)."""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.config import EnvConfig
from env.uav_env import UAVEnv
from visualize import (plot_reward_curve, plot_path,
                       plot_loss_curve, plot_success_curve,
                       make_flight_gif)
from ddqn_agent import DDQNAgent

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_log():
    rewards, successes, losses = [], [], []
    with open(os.path.join(RESULTS, "training_log.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
            successes.append(int(row["success"]))
            v = row["ep_avg_loss"]
            if v and v.lower() != "nan":
                losses.append(float(v))
    return rewards, successes, losses


def evaluate(config, agent, n_episodes=10):
    agent.epsilon = 0.0
    best_reward = -float("inf")
    best_env = None
    successes = 0
    total_rewards = []

    for _ in range(n_episodes):
        env = UAVEnv(config)
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        total_rewards.append(ep_reward)
        if info.get("event") == "mission_complete":
            successes += 1
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_env = env

    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    print(f"Avg Reward : {np.mean(total_rewards):.1f}")
    print(f"Success    : {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")
    if best_env:
        best_env.render()
        print(f"Path length: {len(best_env.path)}")
    return best_env


if __name__ == "__main__":
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
        state_dim=obs_dim, action_dim=act_dim,
        lr=3e-4, gamma=0.99,
        epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9985,
        batch_size=128, buffer_capacity=250000,
        target_update_freq=1000, hidden_dim=256,
        grad_clip=5.0,
    )
    agent.load(os.path.join(RESULTS, "ddqn_model.pt"))
    agent.epsilon = 0.0

    # ── Training history plots ──
    rewards, successes, losses = load_log()
    plot_reward_curve(rewards, window=50, title="DDQN Training Curve",
                      save_path=os.path.join(RESULTS, "training_curve.png"))
    plot_success_curve(successes, window=50, title="DDQN Success Rate",
                       save_path=os.path.join(RESULTS, "success_curve.png"))
    if losses:
        plot_loss_curve(losses, window=50, title="DDQN Training Loss",
                        save_path=os.path.join(RESULTS, "loss_curve.png"))

    # ── Evaluate ──
    best_env = evaluate(config, agent, n_episodes=10)

    # ── Path & GIF ──
    if best_env:
        plot_path(best_env, title="DDQN - Best Path",
                  save_path=os.path.join(RESULTS, "best_path.png"))
        make_flight_gif(best_env, title="DDQN Flight",
                        save_path=os.path.join(RESULTS, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {RESULTS} ===")
