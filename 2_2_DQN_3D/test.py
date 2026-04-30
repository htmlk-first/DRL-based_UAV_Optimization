"""DQN 3D Test: loads saved model, regenerates result PNGs (no training)."""
import os, sys
import numpy as np
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.experiment import (ExperimentPaths, optional_float,
                               print_eval_summary, read_training_log,
                               success_from_info)
from env.config import EnvConfig3D
from env.uav_env import UAVEnv3D
from visualize import (plot_reward_curve, plot_path_3d,
                       plot_loss_curve, plot_success_curve,
                       make_flight_gif_3d)
from dqn_agent import DQNAgent

RESULTS = str(ExperimentPaths.from_file(__file__).results_dir)


def load_log():
    rewards, successes, losses = [], [], []
    for row in read_training_log(RESULTS):
        rewards.append(float(row["reward"]))
        successes.append(int(row["success"]))
        v = optional_float(row, "ep_avg_loss")
        if v is not None:
            losses.append(v)
    return rewards, successes, losses


def evaluate(config, agent, n_episodes=10):
    agent.epsilon = 0.0
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
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        total_rewards.append(ep_reward)
        successes += success_from_info(info)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_env = env

    print_eval_summary(total_rewards, successes, n_episodes,
                       reward_label="Avg Reward ", success_label="Success   ")
    if best_env:
        best_env.render()
        print(f"Path length: {len(best_env.path)}")
    return best_env


if __name__ == "__main__":
    config = EnvConfig3D(
        grid_size_x=20, grid_size_y=20, grid_size_z=5,
        obstacle_mode="fixed",
        energy_budget_multiplier=2.0,
    )

    tmp_env = UAVEnv3D(config)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n

    agent = DQNAgent(
        state_dim=obs_dim, action_dim=act_dim,
        lr=5e-4, gamma=0.99,
        epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.998,
        batch_size=64, buffer_capacity=100000,
        target_update_freq=500, hidden_dim=256,
    )
    agent.load(os.path.join(RESULTS, "dqn_3d_model.pt"))
    agent.epsilon = 0.0

    # ── Training history plots ──
    rewards, successes, losses = load_log()
    plot_reward_curve(rewards, window=50, title="DQN 3D Training Curve",
                      save_path=os.path.join(RESULTS, "training_curve.png"))
    plot_success_curve(successes, window=50, title="DQN 3D Success Rate",
                       save_path=os.path.join(RESULTS, "success_curve.png"))
    if losses:
        plot_loss_curve(losses, window=50, title="DQN 3D Training Loss",
                        save_path=os.path.join(RESULTS, "loss_curve.png"))

    # ── Evaluate ──
    best_env = evaluate(config, agent, n_episodes=10)

    # ── Path & GIF ──
    if best_env:
        plot_path_3d(best_env, title="DQN 3D - Best Path",
                     save_path=os.path.join(RESULTS, "best_path_3d.png"))
        make_flight_gif_3d(best_env, title="DQN 3D Flight",
                           save_path=os.path.join(RESULTS, "flight_3d.gif"), fps=5)

    print(f"\n=== Results saved to {RESULTS} ===")
