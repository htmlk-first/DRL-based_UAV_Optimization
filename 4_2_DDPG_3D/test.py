"""DDPG 3D Test: loads saved model, regenerates result PNGs (no training)."""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.config import EnvConfig3D
from env.uav_env import UAVEnv3D
from visualize import (plot_reward_curve, plot_path_3d,
                       plot_critic_loss, plot_actor_loss,
                       plot_success_curve, make_flight_gif_3d)
from ddpg_agent import DDPGAgent

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_log():
    rewards, successes, critic_losses, actor_losses = [], [], [], []
    with open(os.path.join(RESULTS, "training_log.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
            successes.append(int(row["success"]))
            cl = row["ep_avg_critic_loss"]
            al = row["ep_avg_actor_loss"]
            if cl and cl.lower() != "nan":
                critic_losses.append(float(cl))
            if al and al.lower() != "nan":
                actor_losses.append(float(al))
    return rewards, successes, critic_losses, actor_losses


def evaluate(config, agent, n_episodes=10):
    best_reward = -float("inf")
    best_env = None
    successes = 0
    total_rewards = []

    for _ in range(n_episodes):
        env = UAVEnv3D(config)
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, add_noise=False)
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
    config = EnvConfig3D(
        grid_size_x=30, grid_size_y=30, grid_size_z=5,
        obstacle_mode="fixed",
        num_buildings=40,
        energy_budget_multiplier=3.0,
        max_step_size=1.5, max_step_size_z=1.0,
        wp_reach_radius=1.0,
        z_cost_multiplier=1.5,
    )

    tmp_env = UAVEnv3D(config)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]

    agent = DDPGAgent(
        state_dim=obs_dim, action_dim=act_dim,
        actor_lr=1e-4, critic_lr=5e-4,
        gamma=0.99, tau=0.005,
        batch_size=128, buffer_capacity=100000,
        hidden_dim=256,
        noise_sigma=0.3, noise_sigma_min=0.01, noise_decay=0.9972,
        grad_clip=1.0,
    )
    agent.load(os.path.join(RESULTS, "ddpg_3d_model.pt"))

    # ── Training history plots ──
    rewards, successes, c_losses, a_losses = load_log()
    plot_reward_curve(rewards, window=50, title="DDPG 3D Training Curve",
                      save_path=os.path.join(RESULTS, "training_curve.png"))
    plot_success_curve(successes, window=50, title="DDPG 3D Success Rate",
                       save_path=os.path.join(RESULTS, "success_curve.png"))
    if c_losses:
        plot_critic_loss(c_losses, window=50, title="DDPG 3D Critic Loss",
                         save_path=os.path.join(RESULTS, "critic_loss.png"))
    if a_losses:
        plot_actor_loss(a_losses, window=50, title="DDPG 3D Actor Loss",
                        save_path=os.path.join(RESULTS, "actor_loss.png"))

    # ── Evaluate ──
    best_env = evaluate(config, agent, n_episodes=10)

    # ── Path & GIF ──
    if best_env:
        plot_path_3d(best_env, title="DDPG 3D - Best Path",
                     save_path=os.path.join(RESULTS, "best_path_3d.png"))
        make_flight_gif_3d(best_env, title="DDPG 3D Flight",
                           save_path=os.path.join(RESULTS, "flight_3d.gif"), fps=5)

    print(f"\n=== Results saved to {RESULTS} ===")
