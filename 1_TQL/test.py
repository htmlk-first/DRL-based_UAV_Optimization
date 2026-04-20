"""TQL Test: loads saved model, regenerates result PNGs (no training)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.config import EnvConfig
from env.uav_env import UAVEnv
from visualize import plot_path, plot_qvalue_heatmap
from tql_agent import TQLAgent

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def evaluate(config, agent, n_episodes=10):
    agent.epsilon = 0.0
    best_reward = -float("inf")
    best_env = None
    successes = 0
    total_rewards = []

    for _ in range(n_episodes):
        env = UAVEnv(config)
        env.reset()
        state = env.get_state_tuple()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = env.get_state_tuple()
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
        grid_size=10,
        obstacle_mode="fixed",
        energy_budget_multiplier=1.5,
    )

    agent = TQLAgent(
        n_actions=4,
        lr=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
    )
    agent.load(os.path.join(RESULTS, "tql_model.pkl"))

    # ── Evaluate ──
    best_env = evaluate(config, agent, n_episodes=10)

    # ── Q-value heatmaps ──
    tmp_env = UAVEnv(config)
    tmp_env.reset()
    for wp_i in range(tmp_env.n_waypoints):
        plot_qvalue_heatmap(
            agent, tmp_env,
            waypoint_idx=wp_i,
            title="TQL Q-Value Heatmap",
            save_path=os.path.join(RESULTS, f"qvalue_heatmap_wp{wp_i}.png"),
        )

    # ── Path ──
    if best_env:
        plot_path(best_env, title="TQL - Best Path",
                  save_path=os.path.join(RESULTS, "best_path.png"))

    try:
        from visualize import make_flight_gif
        if best_env:
            make_flight_gif(best_env, title="TQL Flight",
                            save_path=os.path.join(RESULTS, "flight.gif"), fps=5)
    except ImportError:
        pass

    print(f"\n=== Results saved to {RESULTS} ===")
