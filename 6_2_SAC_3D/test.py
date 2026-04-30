"""SAC 3D Test: loads saved model, regenerates result PNGs (no training)."""
import os, sys, copy
import numpy as np
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.experiment import (ExperimentPaths, optional_float,
                               print_eval_summary, read_training_log,
                               select_existing_result, success_from_info)
from env import UAVEnv3D, EnvConfig3D
from visualize import (plot_reward_curve, plot_success_curve,
                       plot_actor_loss, plot_critic_loss, plot_alpha_curve,
                       plot_path_3d, make_flight_gif_3d)
from sac_agent import SACAgent

RESULTS = str(ExperimentPaths.from_file(__file__).results_dir)


def load_log():
    rewards, successes = [], []
    actor_losses, critic_losses, alphas = [], [], []
    for row in read_training_log(RESULTS):
        rewards.append(float(row["reward"]))
        successes.append(int(row["success"]))
        alphas.append(float(row["alpha"]))
        al = optional_float(row, "ep_avg_actor_loss")
        cl = optional_float(row, "ep_avg_critic_loss")
        if al is not None:
            actor_losses.append(al)
        if cl is not None:
            critic_losses.append(cl)
    return rewards, successes, actor_losses, critic_losses, alphas


def evaluate(config, agent, n_episodes=10):
    best_reward = -float("inf")
    best_path = None
    best_env = None
    successes = 0
    total_rewards = []

    for _ in range(n_episodes):
        env = UAVEnv3D(config=config)
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
        successes += success_from_info(info)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = list(env.path)
            best_env = copy.deepcopy(env)

    if successes == 0:
        print("[Deterministic eval failed, retrying stochastic...]")
        total_rewards, successes, best_reward, best_path, best_env = [], 0, -float("inf"), None, None
        for _ in range(n_episodes):
            env = UAVEnv3D(config=config)
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

    print_eval_summary(total_rewards, successes, n_episodes,
                       reward_label="Avg Reward ", success_label="Success   ")
    if best_path:
        print(f"Path length: {len(best_path)}")
    return best_path, best_env


if __name__ == "__main__":
    config = EnvConfig3D(
        grid_size_x=100, grid_size_y=100, grid_size_z=10,
        obstacle_mode="fixed",
        building_footprint_size=3,
        num_buildings=50,
        energy_budget_multiplier=3.6,
        max_step_size=2.2, max_step_size_z=1.0,
        wp_reach_radius=2.0,
        z_cost_multiplier=1.7,
        penalty_z_reversal=-1.0,
        max_steps=1500,
    )

    env = UAVEnv3D(config=config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=256,
        lr_actor=1.5e-4, lr_critic=3e-4, lr_alpha=1e-4,
        gamma=0.995, tau=0.005,
        buffer_capacity=750000, batch_size=256,
        initial_alpha=0.2,
        grad_clip=1.0,
    )
    model_path = select_existing_result(RESULTS, "sac_3d_best_model.pt", "sac_3d_model.pt")
    print(f"Loading model: {model_path}")
    agent.load(model_path)

    # ── Training history plots ──
    rewards, successes, a_losses, c_losses, alphas = load_log()
    plot_reward_curve(rewards, window=50, title="SAC 3D Training Curve",
                      save_path=os.path.join(RESULTS, "training_curve.png"))
    plot_success_curve(successes, window=50, title="SAC 3D Success Rate",
                       save_path=os.path.join(RESULTS, "success_curve.png"))
    if a_losses:
        plot_actor_loss(a_losses, window=50, title="SAC 3D Actor Loss",
                        save_path=os.path.join(RESULTS, "actor_loss.png"))
    if c_losses:
        plot_critic_loss(c_losses, window=50, title="SAC 3D Critic Loss",
                         save_path=os.path.join(RESULTS, "critic_loss.png"))
    if alphas:
        plot_alpha_curve(alphas, window=50, title="SAC 3D Entropy Coefficient (α)",
                         save_path=os.path.join(RESULTS, "alpha_curve.png"))

    # ── Evaluate ──
    best_path, best_env = evaluate(config, agent, n_episodes=10)

    # ── Path & GIF ──
    if best_env and best_path:
        plot_path_3d(best_env, title="SAC 3D - Best Path",
                     save_path=os.path.join(RESULTS, "best_path_3d.png"))
        make_flight_gif_3d(best_env, title="SAC 3D Flight",
                           save_path=os.path.join(RESULTS, "flight_3d.gif"), fps=5)

    print(f"\n=== Results saved to {RESULTS} ===")
