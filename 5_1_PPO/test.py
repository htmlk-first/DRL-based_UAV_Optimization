"""PPO Test: loads saved model, regenerates result PNGs (no training)."""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.config import EnvConfig
from env.uav_env import UAVEnv
from visualize import (plot_reward_curve, plot_path,
                       plot_policy_loss, plot_value_loss,
                       plot_entropy_curve,
                       plot_success_curve, make_flight_gif)
from ppo_agent import PPOAgent

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_log():
    rewards, successes = [], []
    policy_losses, value_losses, entropies = [], [], []
    with open(os.path.join(RESULTS, "training_log.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
            successes.append(int(row["success"]))
            for src, dst in [("policy_loss", policy_losses),
                              ("value_loss",  value_losses),
                              ("entropy",     entropies)]:
                v = row[src]
                if v and v.lower() != "nan":
                    dst.append(float(v))
    return rewards, successes, policy_losses, value_losses, entropies


def evaluate(config, agent, n_episodes=10):
    best_reward = -float("inf")
    best_env = None
    successes = 0
    total_rewards = []

    # deterministic pass
    for _ in range(n_episodes):
        env = UAVEnv(config)
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
        if info.get("event") == "mission_complete":
            successes += 1
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_env = env

    # stochastic fallback
    if successes == 0:
        print("[Deterministic eval failed, retrying stochastic...]")
        total_rewards, successes, best_reward, best_env = [], 0, -float("inf"), None
        for _ in range(n_episodes):
            env = UAVEnv(config)
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, _, _ = agent.select_action(obs, deterministic=False)
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
        grid_size=30,
        obstacle_mode="fixed",
        energy_budget_multiplier=3.0,
        max_step_size=1.5,
        wp_reach_radius=1.0,
    )

    tmp_env = UAVEnv(config)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=obs_dim, action_dim=act_dim,
        lr=3e-4, gamma=0.99,
        gae_lambda=0.95, clip_epsilon=0.2,
        entropy_coeff=0.01, value_coeff=0.25,
        max_grad_norm=0.5, k_epochs=10,
        mini_batch_size=64, hidden_dim=256,
        lr_decay=False,
    )
    agent.load(os.path.join(RESULTS, "ppo_model.pt"))

    # ── Training history plots ──
    rewards, successes, p_losses, v_losses, ents = load_log()
    plot_reward_curve(rewards, window=50, title="PPO Training Curve",
                      save_path=os.path.join(RESULTS, "training_curve.png"))
    plot_success_curve(successes, window=50, title="PPO Success Rate",
                       save_path=os.path.join(RESULTS, "success_curve.png"))
    if p_losses:
        plot_policy_loss(p_losses, window=50, title="PPO Policy Loss",
                         save_path=os.path.join(RESULTS, "policy_loss.png"))
    if v_losses:
        plot_value_loss(v_losses, window=50, title="PPO Value Loss",
                        save_path=os.path.join(RESULTS, "value_loss.png"))
    if ents:
        plot_entropy_curve(ents, window=50, title="PPO Entropy",
                           save_path=os.path.join(RESULTS, "entropy.png"))

    # ── Evaluate ──
    best_env = evaluate(config, agent, n_episodes=10)

    # ── Path & GIF ──
    if best_env:
        plot_path(best_env, title="PPO - Best Path",
                  save_path=os.path.join(RESULTS, "best_path.png"))
        make_flight_gif(best_env, title="PPO Flight",
                        save_path=os.path.join(RESULTS, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {RESULTS} ===")
