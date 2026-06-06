"""MADDPG Test: loads saved model and regenerates result plots."""

from __future__ import annotations

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common.experiment import (
    ExperimentPaths,
    optional_float,
    print_eval_summary,
    read_training_log,
    success_from_info,
)
from env.config import MultiAgentEnvConfig
from env.multi_uav_env import MultiUAVEnv
from maddpg_agent import MADDPGAgent
from visualize import (
    make_flight_gif,
    plot_actor_loss,
    plot_critic_loss,
    plot_path,
    plot_reward_curve,
    plot_success_curve,
)

RESULTS = str(ExperimentPaths.from_file(__file__).results_dir)


def build_config():
    return MultiAgentEnvConfig(
        n_agents=4,
        start_positions=[(0.0, 0.0), (0.0, 19.0), (19.0, 0.0), (19.0, 19.0)],
        waypoints=[
            (4.0, 4.0), (4.0, 10.0), (4.0, 15.0),
            (8.0, 6.0), (8.0, 13.0),
            (12.0, 4.0), (12.0, 10.0), (12.0, 16.0),
            (16.0, 7.0), (16.0, 14.0),
        ],
        grid_size=20,
        obstacle_mode="fixed",
        ordered_waypoints=False,
        energy_budget_multiplier=2.4,
        max_step_size=1.0,
        wp_reach_radius=0.9,
        collision_radius=0.6,
        collision_penalty=-20.0,
        max_steps=1000,
    )


def load_log():
    rewards, successes, critic_losses, actor_losses = [], [], [], []
    for row in read_training_log(RESULTS):
        rewards.append(float(row["reward"]))
        successes.append(int(row["success"]))
        critic_loss = optional_float(row, "ep_avg_critic_loss")
        actor_loss = optional_float(row, "ep_avg_actor_loss")
        if critic_loss is not None:
            critic_losses.append(critic_loss)
        if actor_loss is not None:
            actor_losses.append(actor_loss)
    return rewards, successes, critic_losses, actor_losses


def evaluate(config, agent, n_episodes=10):
    best_reward = -float("inf")
    best_env = None
    successes = 0
    total_rewards = []

    for _ in range(n_episodes):
        env = MultiUAVEnv(config)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info = {}

        while not done:
            actions = agent.select_actions(obs, add_noise=False)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            ep_reward += float(np.mean(rewards))

        total_rewards.append(ep_reward)
        successes += success_from_info(info)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_env = env

    print_eval_summary(total_rewards, successes, n_episodes,
                       reward_label="Avg Team Return", success_label="Success Rate")
    if best_env is not None:
        best_env.render()
        print(f"Path lengths: {[len(path) for path in best_env.paths]}")
    return best_env


if __name__ == "__main__":
    config = build_config()
    tmp_env = MultiUAVEnv(config)
    agent = MADDPGAgent(
        obs_dim=tmp_env.single_observation_dim,
        action_dim=tmp_env.action_space.shape[-1],
        n_agents=config.n_agents,
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_capacity=600000,
        hidden_dim=256,
        noise_sigma=0.35,
        noise_sigma_min=0.03,
        noise_decay=0.9992,
        grad_clip=1.0,
        reward_scale=0.01,
        warmup_steps=5000,
        learn_every=4,
        policy_delay=2,
        action_l2=1e-3,
        bc_weight=1.0,
        bc_weight_min=0.05,
        bc_decay_steps=100000,
        guidance_mix=0.5,
        guidance_decay_steps=50000,
    )

    model_path = os.path.join(RESULTS, "maddpg_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run train.py before test.py."
        )
    try:
        agent.load(model_path)
    except RuntimeError as exc:
        raise RuntimeError(
            "The saved maddpg_model.pt is incompatible with the current "
            "4-UAV / 10-waypoint MADDPG observation/action dimensions. "
            "Run train.py again to create a fresh checkpoint."
        ) from exc

    rewards, successes, critic_losses, actor_losses = load_log()
    plot_reward_curve(rewards, window=100, title="MADDPG Training Curve",
                      save_path=os.path.join(RESULTS, "training_curve.png"))
    plot_success_curve(successes, window=100, title="MADDPG Success Rate",
                       save_path=os.path.join(RESULTS, "success_curve.png"))
    if critic_losses:
        plot_critic_loss(critic_losses, window=200, title="MADDPG Critic Loss",
                         save_path=os.path.join(RESULTS, "critic_loss.png"))
    if actor_losses:
        plot_actor_loss(actor_losses, window=200, title="MADDPG Actor Loss",
                        save_path=os.path.join(RESULTS, "actor_loss.png"))

    best_env = evaluate(config, agent, n_episodes=10)
    if best_env is not None:
        plot_path(best_env, title="MADDPG - Best Cooperative Path",
                  save_path=os.path.join(RESULTS, "best_path.png"))
        make_flight_gif(best_env, title="MADDPG Cooperative Flight",
                        save_path=os.path.join(RESULTS, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {RESULTS} ===")
