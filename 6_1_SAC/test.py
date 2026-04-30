"""SAC Test: loads saved model, regenerates result PNGs (no training)."""
import os, sys, csv, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import UAVEnv, EnvConfig
from visualize import (plot_reward_curve, plot_success_curve,
                       plot_actor_loss, plot_critic_loss, plot_alpha_curve,
                       plot_path, make_flight_gif)
from sac_agent import SACAgent

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_log():
    rewards, successes = [], []
    actor_losses, critic_losses, alphas = [], [], []
    with open(os.path.join(RESULTS, "training_log.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))
            successes.append(int(row["success"]))
            alphas.append(float(row["alpha"]))
            al = row["ep_avg_actor_loss"]
            cl = row["ep_avg_critic_loss"]
            if al and al.lower() != "nan":
                actor_losses.append(float(al))
            if cl and cl.lower() != "nan":
                critic_losses.append(float(cl))
    return rewards, successes, actor_losses, critic_losses, alphas


def evaluate(config, agent, n_episodes=10):
    best_reward = -float("inf")
    best_path = None
    best_env = None
    successes = 0
    total_rewards = []

    for _ in range(n_episodes):
        env = UAVEnv(config=config)
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
        if info.get("event") == "mission_complete":
            successes += 1
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = list(env.path)
            best_env = copy.deepcopy(env)

    if successes == 0:
        print("[Deterministic eval failed, retrying stochastic...]")
        total_rewards, successes, best_reward, best_path, best_env = [], 0, -float("inf"), None, None
        for _ in range(n_episodes):
            env = UAVEnv(config=config)
            state, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = agent.select_action(state, deterministic=False)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
            total_rewards.append(ep_reward)
            if ep_reward > best_reward:
                best_reward = ep_reward
                best_path = list(env.path)
                best_env = copy.deepcopy(env)

    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    print(f"Avg Reward : {np.mean(total_rewards):.1f}")
    print(f"Success    : {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")
    if best_path:
        print(f"Path length: {len(best_path)}")
    return best_path, best_env


if __name__ == "__main__":
    config = EnvConfig(
        grid_size=100,
        obstacle_mode="fixed",
        obstacle_footprint_size=3,
        energy_budget_multiplier=3.0,
        max_step_size=2.2,
        wp_reach_radius=1.8,
    )

    env = UAVEnv(config=config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=256,
        lr_actor=2e-4, lr_critic=3e-4, lr_alpha=1e-4,
        gamma=0.995, tau=0.005,
        buffer_capacity=500000, batch_size=256,
        initial_alpha=0.2,
        grad_clip=1.0,
    )
    model_path = os.path.join(RESULTS, "sac_best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(RESULTS, "sac_model.pt")
    print(f"Loading model: {model_path}")
    agent.load(model_path)

    # ── Training history plots ──
    rewards, successes, a_losses, c_losses, alphas = load_log()
    plot_reward_curve(rewards, window=50, title="SAC Training Curve",
                      save_path=os.path.join(RESULTS, "training_curve.png"))
    plot_success_curve(successes, window=50, title="SAC Success Rate",
                       save_path=os.path.join(RESULTS, "success_curve.png"))
    if a_losses:
        plot_actor_loss(a_losses, window=50, title="SAC Actor Loss",
                        save_path=os.path.join(RESULTS, "actor_loss.png"))
    if c_losses:
        plot_critic_loss(c_losses, window=50, title="SAC Critic Loss",
                         save_path=os.path.join(RESULTS, "critic_loss.png"))
    if alphas:
        plot_alpha_curve(alphas, window=50, title="SAC Entropy Coefficient (α)",
                         save_path=os.path.join(RESULTS, "alpha_curve.png"))

    # ── Evaluate ──
    best_path, best_env = evaluate(config, agent, n_episodes=10)

    # ── Path & GIF ──
    if best_env and best_path:
        plot_path(best_env, best_path, title="SAC - Best Path",
                  save_path=os.path.join(RESULTS, "best_path.png"))
        make_flight_gif(best_env, best_path, title="SAC Flight",
                        save_path=os.path.join(RESULTS, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {RESULTS} ===")
