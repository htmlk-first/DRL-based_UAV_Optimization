"""Visualization helpers for cooperative multi-UAV Double DQN experiments."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle


BG = "#f8fafc"
GRID = "#cbd5e1"
OBSTACLE = "#334155"
WAYPOINT = "#f59e0b"
VISITED = "#22c55e"
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]


def _ensure_dir(path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _moving_average(values, window):
    if len(values) < window:
        return None, None
    ma = np.convolve(values, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(values) + 1)
    return x, ma


def plot_reward_curve(rewards, window=100, title="MADDQN Training Curve", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    eps = np.arange(1, len(rewards) + 1)
    ax.plot(eps, rewards, color=COLORS[0], alpha=0.25, linewidth=0.8, label="episode")
    x_ma, ma = _moving_average(np.asarray(rewards, dtype=float), window)
    if ma is not None:
        ax.plot(x_ma, ma, color=COLORS[0], linewidth=2.0, label=f"MA{window}")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Team return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


def plot_success_curve(success_history, window=100, title="MADDQN Success Rate", save_path=None):
    values = np.asarray(success_history, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    eps = np.arange(1, len(values) + 1)
    ax.plot(eps, values * 100, color=VISITED, alpha=0.25, linewidth=0.8, label="episode")
    x_ma, ma = _moving_average(values, window)
    if ma is not None:
        ax.plot(x_ma, ma * 100, color=VISITED, linewidth=2.0, label=f"MA{window}")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


def plot_loss_curve(losses, window=200, title="MADDQN Training Loss", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, color=COLORS[1], alpha=0.2, linewidth=0.7, label="loss")
    x_ma, ma = _moving_average(np.asarray(losses, dtype=float), window)
    if ma is not None:
        ax.plot(x_ma, ma, color=COLORS[1], linewidth=2.0, label=f"MA{window}")
    ax.set_title(title)
    ax.set_xlabel("Learning step")
    ax.set_ylabel("Huber loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


def _draw_static(ax, env, title):
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(env.grid_size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.35, alpha=0.6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(title)

    for x, y in env.obstacles:
        ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, color=OBSTACLE, alpha=0.9))

    for i, (x, y) in enumerate(env.waypoints):
        color = VISITED if env.visited[i] else WAYPOINT
        ax.scatter(y, x, s=150, marker="*", color=color, edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(y, x + 0.38, f"W{i}", ha="center", va="center", fontsize=8, color="#0f172a")


def plot_path(env, title="MADDQN - Best Cooperative Path", save_path=None):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=140)
    _draw_static(ax, env, title)

    for agent_idx, path in enumerate(env.paths):
        if not path:
            continue
        color = COLORS[agent_idx % len(COLORS)]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(ys, xs, color=color, linewidth=2.0, alpha=0.9, label=f"UAV {agent_idx}")
        ax.scatter(ys[0], xs[0], s=85, marker="s", color=color, edgecolor="white", zorder=5)
        ax.scatter(ys[-1], xs[-1], s=110, marker="o", color=color, edgecolor="white", zorder=6)

    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


def make_flight_gif(env, title="MADDQN Cooperative Flight", save_path="flight.gif", fps=5):
    if not env.paths or not env.paths[0]:
        print("  [make_flight_gif] path is empty, skipping.")
        return

    n_frames = max(len(path) for path in env.paths)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

    def draw_frame(frame_idx):
        ax.clear()
        _draw_static(ax, env, f"{title} | Step {frame_idx}/{n_frames - 1}")
        for agent_idx, path in enumerate(env.paths):
            color = COLORS[agent_idx % len(COLORS)]
            idx = min(frame_idx, len(path) - 1)
            trail = path[:idx + 1]
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            ax.plot(ys, xs, color=color, linewidth=2.0, alpha=0.8)
            ax.scatter(ys[-1], xs[-1], s=130, marker="o", color=color, edgecolor="white", zorder=6)
            ax.text(ys[-1], xs[-1] - 0.45, f"A{agent_idx}", ha="center", fontsize=9, color=color)

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 / fps)
    _ensure_dir(save_path)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
