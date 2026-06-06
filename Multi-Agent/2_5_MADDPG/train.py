"""Train cooperative MADDPG on the 2D UAV waypoint task."""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common.experiment import ExperimentPaths, print_eval_summary, success_from_info
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


def team_rewards(rewards):
    """Broadcast the fully cooperative team reward to all critics."""
    reward = float(np.mean(rewards))
    return np.full(len(rewards), reward, dtype=np.float32)


def rollout_episode(config, agent, add_noise=False):
    env = MultiUAVEnv(config)
    obs, _ = env.reset()
    if add_noise:
        agent.reset_noise()
    done = False
    episode_return = 0.0
    info = {}

    while not done:
        actions = agent.select_actions(obs, add_noise=add_noise)
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        episode_return += float(np.mean(rewards))

    return episode_return, success_from_info(info), info, env


def train(config, agent, n_episodes=5000, print_every=250, log_path=None):
    env = MultiUAVEnv(config)
    rewards_history = []
    success_history = []
    critic_loss_history = []
    actor_loss_history = []
    collision_history = []
    best_eval_score = (-1, -1, -float("inf"))
    best_eval_return = -float("inf")
    best_success_path = None
    best_return_path = None

    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        results_dir = os.path.dirname(os.path.abspath(log_path))
        best_success_path = os.path.join(
            results_dir, "maddpg_best_success.pt"
        )
        best_return_path = os.path.join(
            results_dir, "maddpg_best_return.pt"
        )
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "episode",
            "reward",
            "avg_team_return",
            "success",
            "noise_sigma",
            "ep_avg_critic_loss",
            "ep_avg_actor_loss",
            "buffer_size",
            "steps",
            "collisions",
            "waypoints_visited",
            "greedy_reward",
            "greedy_success",
            "greedy_waypoints_visited",
        ])

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        agent.reset_noise()
        done = False
        ep_team_return = 0.0
        ep_critic_losses = []
        ep_actor_losses = []
        info = {}

        while not done:
            guidance_actions = env.get_guided_actions()
            actions = agent.select_actions(obs, add_noise=True)
            actions = agent.blend_guidance(actions, guidance_actions)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            agent.store_joint(
                obs,
                actions,
                team_rewards(rewards),
                next_obs,
                done,
                guidance_actions,
            )
            critic_loss, actor_loss = agent.learn()
            if critic_loss is not None:
                critic_loss_history.append(critic_loss)
                ep_critic_losses.append(critic_loss)
            if actor_loss is not None:
                actor_loss_history.append(actor_loss)
                ep_actor_losses.append(actor_loss)

            obs = next_obs
            ep_team_return += float(np.mean(rewards))

        agent.decay_noise()
        success = success_from_info(info)
        rewards_history.append(ep_team_return)
        success_history.append(success)
        collision_history.append(info.get("collisions", 0))

        greedy_reward = float("nan")
        greedy_success = ""
        greedy_waypoints = ""
        should_evaluate = ep % print_every == 0 or ep == n_episodes
        if should_evaluate:
            (
                greedy_reward,
                greedy_success,
                greedy_info,
                _,
            ) = rollout_episode(config, agent, add_noise=False)
            greedy_waypoints = greedy_info.get("waypoints_visited", 0)
            eval_score = (
                int(greedy_success),
                int(greedy_waypoints),
                float(greedy_reward),
            )
            if best_success_path is not None and eval_score > best_eval_score:
                best_eval_score = eval_score
                agent.save(best_success_path)
            if (
                best_return_path is not None
                and greedy_reward > best_eval_return
            ):
                best_eval_return = greedy_reward
                agent.save(best_return_path)

        if csv_writer is not None:
            ep_avg_cl = float(np.mean(ep_critic_losses)) if ep_critic_losses else float("nan")
            ep_avg_al = float(np.mean(ep_actor_losses)) if ep_actor_losses else float("nan")
            csv_writer.writerow([
                ep,
                ep_team_return,
                ep_team_return,
                success,
                agent.noise.sigma,
                ep_avg_cl,
                ep_avg_al,
                len(agent.buffer),
                info.get("steps", 0),
                info.get("collisions", 0),
                info.get("waypoints_visited", 0),
                greedy_reward,
                greedy_success,
                greedy_waypoints,
            ])
            csv_file.flush()

        if should_evaluate:
            avg_r = float(np.mean(rewards_history[-print_every:]))
            avg_s = float(np.mean(success_history[-print_every:]) * 100)
            avg_cl = float(np.mean(critic_loss_history[-1000:])) if critic_loss_history else 0.0
            avg_al = float(np.mean(actor_loss_history[-1000:])) if actor_loss_history else 0.0
            avg_c = float(np.mean(collision_history[-print_every:]))
            print(f"[Ep {ep:5d}] AvgTeamReturn: {avg_r:8.1f} | "
                  f"Success: {avg_s:5.1f}% | "
                  f"Greedy: {int(greedy_success)}/"
                  f"{int(greedy_waypoints)} WP, {greedy_reward:7.1f} | "
                  f"NoiseSigma: {agent.noise.sigma:.4f} | "
                  f"CriticL: {avg_cl:.4f} | ActorL: {avg_al:.4f} | "
                  f"AvgCollisions: {avg_c:.1f} | "
                  f"Buffer: {len(agent.buffer)}")

    if csv_file is not None:
        csv_file.close()

    return rewards_history, success_history, critic_loss_history, actor_loss_history


def evaluate(config, agent, n_episodes=10):
    best_reward = -float("inf")
    best_env = None
    total_rewards = []
    successes = 0

    for _ in range(n_episodes):
        ep_reward, success, info, env = rollout_episode(
            config, agent, add_noise=False
        )

        total_rewards.append(ep_reward)
        successes += success
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
    config = MultiAgentEnvConfig(
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

    tmp_env = MultiUAVEnv(config)
    obs_dim = tmp_env.single_observation_dim
    act_dim = tmp_env.action_space.shape[-1]

    agent = MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=act_dim,
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

    paths = ExperimentPaths.from_file(__file__)
    save_dir = str(paths.results_dir)

    print("=== MADDPG Training Start ===")
    print(f"Agents: {config.n_agents}, State dim: {obs_dim}, Action dim: {act_dim}")
    print("Training: centralized critics, Execution: decentralized actors")
    print(f"Max step size: {config.max_step_size}, WP radius: {config.wp_reach_radius}")
    print(
        f"Reward scale: {agent.reward_scale} | "
        f"Warm-up: {agent.warmup_steps} | "
        f"Learn every: {agent.learn_every} | "
        f"Policy delay: {agent.policy_delay} | "
        f"Guidance mix: {agent.guidance_mix}"
    )
    print(f"Device: {agent.device}")

    rewards, successes, critic_losses, actor_losses = train(
        config,
        agent,
        n_episodes=5000,
        print_every=250,
        log_path=os.path.join(save_dir, "training_log.csv"),
    )

    final_model_path = os.path.join(save_dir, "maddpg_final.pt")
    best_model_path = os.path.join(save_dir, "maddpg_best_success.pt")
    canonical_model_path = os.path.join(save_dir, "maddpg_model.pt")
    agent.save(final_model_path)
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
    agent.save(canonical_model_path)

    plot_reward_curve(rewards, window=100, title="MADDPG Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))
    plot_success_curve(successes, window=100, title="MADDPG Success Rate",
                       save_path=os.path.join(save_dir, "success_curve.png"))
    if critic_losses:
        plot_critic_loss(critic_losses, window=200, title="MADDPG Critic Loss",
                         save_path=os.path.join(save_dir, "critic_loss.png"))
    if actor_losses:
        plot_actor_loss(actor_losses, window=200, title="MADDPG Actor Loss",
                        save_path=os.path.join(save_dir, "actor_loss.png"))

    best_env = evaluate(config, agent, n_episodes=10)
    if best_env is not None:
        plot_path(best_env, title="MADDPG - Best Cooperative Path",
                  save_path=os.path.join(save_dir, "best_path.png"))
        make_flight_gif(best_env, title="MADDPG Cooperative Flight",
                        save_path=os.path.join(save_dir, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {save_dir} ===")
