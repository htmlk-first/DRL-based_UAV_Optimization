"""
UAV 환경 시각화 모듈 (라이트 테마, SAC용)
- plot_grid           : 현재 환경 상태
- plot_path           : 비행 경로 + 통계 패널
- plot_reward_curve   : 학습 보상 곡선
- plot_success_curve  : 성공률 곡선
- plot_actor_loss     : Actor 손실 곡선
- plot_critic_loss    : Critic 손실 곡선
- plot_alpha_curve    : 엔트로피 계수(α) 곡선 (SAC 전용)
- make_flight_gif     : 비행 경로 GIF 애니메이션
- plot_comparison     : 다중 알고리즘 비교
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter


# ── 공통 색상 팔레트 ──────────────────────────────────────────────────────────
DARK_BG    = "#ffffff"
PANEL_BG   = "#f6f8fa"
GRID_LINE  = "#d0d7de"
BORDER_CLR = "#afb8c1"

UAV_COLOR      = "#cf222e"
PATH_START_CLR = "#0969da"
PATH_END_CLR   = "#cf222e"
WP_COLOR       = "#e5534b"
VISITED_CLR    = "#1a7f37"
OBS_COLOR      = "#57606a"
START_COLOR    = "#bf8700"
EMPTY_CLR      = "#f6f8fa"
TEXT_CLR       = "#24292f"
MUTED_CLR      = "#57606a"
ACCENT_YELLOW  = "#9a6700"

MARKER_EDGE    = "#555555"


def _label_scale(size):
    if size >= 80:
        return 1.45
    if size >= 50:
        return 1.25
    return 1.0


def _scaled(base, scale):
    # 16:9 통일 figsize 에 맞춰 동적 폰트도 1.6배 가산
    return int(round(base * scale * 1.6))


def _grid_step(size):
    if size >= 80:
        return 5
    if size >= 50:
        return 2
    return 1


def _grid_indices(size):
    step = _grid_step(size)
    indices = list(range(0, size + 1, step))
    if indices[-1] != size:
        indices.append(size)
    return indices


def _draw_grid_lines(ax, size, zorder=1):
    step = _grid_step(size)
    linewidth = 0.4 if step == 1 else 0.55
    alpha = 0.6 if step == 1 else 0.38
    for i in _grid_indices(size):
        ax.axhline(i - 0.5, color=GRID_LINE,
                   linewidth=linewidth, alpha=alpha, zorder=zorder)
        ax.axvline(i - 0.5, color=GRID_LINE,
                   linewidth=linewidth, alpha=alpha, zorder=zorder)


# ── 통일된 출력 규격 ─────────────────────────────────────────────────────────
_FIG_SIZE      = (16, 9)
_PANEL_RATIOS  = [1.4, 1.0]
_PNG_DPI       = 140
_GIF_DPI       = 100


def _path_figsize(size):
    return _FIG_SIZE


def _gif_figsize(size):
    return _FIG_SIZE


def _label_x(col, size, dx):
    if col >= size - 4:
        return col - dx, "right"
    return col + dx, "left"


def _h2rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=11)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_CLR)


def _style_fig(fig):
    fig.patch.set_facecolor(DARK_BG)


def _path_cmap():
    return LinearSegmentedColormap.from_list("uav_trail",
                                              [PATH_START_CLR, PATH_END_CLR])


def _draw_grid_bg(ax, env):
    size = env.grid_size
    img = np.tile(_h2rgb(EMPTY_CLR), (size, size, 1))
    for obs in env.obstacles:
        img[obs[0], obs[1]] = _h2rgb(OBS_COLOR)

    ax.imshow(img, origin="upper",
              extent=[-0.5, size - 0.5, size - 0.5, -0.5],
              aspect="equal", interpolation="nearest", zorder=0)

    _draw_grid_lines(ax, size, zorder=1)


def _cum_distances(path):
    """경로를 따라 누적 유클리드 거리 계산"""
    dists = [0.0]
    for i in range(len(path) - 1):
        d = np.sqrt((path[i+1][0] - path[i][0])**2
                    + (path[i+1][1] - path[i][1])**2)
        dists.append(dists[-1] + d)
    return dists


# ── 1. plot_grid ──────────────────────────────────────────────────────────────
def plot_grid(env, title="UAV Grid Environment", save_path=None):
    size = env.grid_size
    fig, ax = plt.subplots(figsize=(max(7, size * 0.6), max(7, size * 0.6)))
    _style_fig(fig)
    _style_ax(ax)
    _draw_grid_bg(ax, env)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")

    s = env.config.start_pos
    ax.plot(s[1], s[0], "s", color=START_COLOR, ms=13,
            mec=MARKER_EDGE, mew=1.5, zorder=5)
    ax.text(s[1], s[0], "S", ha="center", va="center",
            fontsize=13, fontweight="bold", color=DARK_BG, zorder=6)

    for i, wp in enumerate(env.waypoints):
        clr = VISITED_CLR if env.visited[i] else WP_COLOR
        ax.plot(wp[1], wp[0], "*", color=clr, ms=16,
                mec=MARKER_EDGE, mew=1, zorder=5)
        ax.text(wp[1], wp[0], str(i + 1), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=6)

    ax.plot(env.uav_pos[1], env.uav_pos[0], "^",
            color=UAV_COLOR, ms=15, mec=MARKER_EDGE, mew=1.5, zorder=7)

    epct = env.energy / env.energy_budget * 100
    ax.set_title(f"{title}  |  Energy {epct:.0f}%  Step {env.steps}",
                 fontsize=17, fontweight="bold", pad=8)

    legend_elements = [
        mpatches.Patch(facecolor=EMPTY_CLR,   edgecolor=GRID_LINE, label="Empty"),
        mpatches.Patch(facecolor=OBS_COLOR,                        label="Obstacle"),
        mpatches.Patch(facecolor=WP_COLOR,                         label="Waypoint"),
        mpatches.Patch(facecolor=VISITED_CLR,                      label="Visited WP"),
        mpatches.Patch(facecolor=UAV_COLOR,                        label="UAV"),
        mpatches.Patch(facecolor=START_COLOR,                      label="Start"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11,
              facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR)
    ax.set_xlabel("Column (Y)")
    ax.set_ylabel("Row (X)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 2. plot_path ──────────────────────────────────────────────────────────────
def plot_path(env, path=None, title="UAV Flight Path (SAC)", save_path=None):
    if path is None:
        path = env.path
    size = env.grid_size
    fs = _label_scale(size)
    cmap = _path_cmap()
    n = len(path)

    fig, (ax, ax_info) = plt.subplots(
        1, 2, figsize=_path_figsize(size),
        gridspec_kw={"width_ratios": _PANEL_RATIOS}
    )
    _style_fig(fig)
    _style_ax(ax)
    ax_info.set_facecolor(PANEL_BG)
    for spine in ax_info.spines.values():
        spine.set_edgecolor(BORDER_CLR)

    _draw_grid_bg(ax, env)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")

    # 경로 그라디언트 (연속 좌표)
    if n > 1:
        for i in range(n - 1):
            t = i / max(n - 2, 1)
            ax.plot([path[i][1], path[i + 1][1]],
                    [path[i][0], path[i + 1][0]],
                    "-", color=cmap(t), linewidth=max(3.0, 2.4 * fs), alpha=0.9,
                    solid_capstyle="round", zorder=3)
        for i, p in enumerate(path):
            t = i / max(n - 1, 1)
            ax.plot(p[1], p[0], "o", color=cmap(t),
                    ms=max(3.5, 2.8 * fs), alpha=0.55, zorder=4)

    # 에너지 추적 (누적 유클리드 거리 기반)
    cum_dist = _cum_distances(path)
    energy_at_wp = {}
    for step_i, pos in enumerate(path):
        for j, wp in enumerate(env.waypoints):
            dist_to_wp = np.sqrt((pos[0] - wp[0])**2 + (pos[1] - wp[1])**2)
            if dist_to_wp <= env.config.wp_reach_radius and j not in energy_at_wp:
                energy_at_wp[j] = env.energy_budget - cum_dist[step_i]

    # 웨이포인트
    label_dx = max(0.8, 0.8 * fs)
    label_dy = max(1.2, 1.0 * fs)
    energy_dy = max(1.5, 1.15 * fs)
    for i, wp in enumerate(env.waypoints):
        clr = VISITED_CLR if env.visited[i] else WP_COLOR
        text_x, text_ha = _label_x(wp[1], size, label_dx)
        wp_label_y = wp[0] - label_dy
        energy_label_y = wp[0] + energy_dy
        if wp[0] >= size - 5:
            wp_label_y = wp[0] - label_dy * 1.8
            energy_label_y = wp[0] - label_dy * 0.6
        ax.plot(wp[1], wp[0], "*", color=clr, ms=_scaled(18, fs),
                mec=MARKER_EDGE, mew=1.2, zorder=6)
        ax.text(wp[1], wp[0], str(i + 1), ha="center", va="center",
                fontsize=_scaled(11, fs), fontweight="bold",
                color="white", zorder=7)
        ax.text(text_x, wp_label_y, f"WP{i + 1}",
                ha=text_ha, fontsize=_scaled(13, fs),
                fontweight="bold", color=clr, zorder=7,
                path_effects=[pe.withStroke(linewidth=max(2.0, 1.8 * fs),
                                            foreground=DARK_BG)])
        if i in energy_at_wp:
            pct = energy_at_wp[i] / env.energy_budget * 100
            ax.text(text_x, energy_label_y,
                    f"E:{pct:.0f}%", fontsize=_scaled(12, fs),
                    ha=text_ha, color=VISITED_CLR, zorder=7,
                    path_effects=[pe.withStroke(linewidth=max(2.0, 1.6 * fs),
                                                foreground=DARK_BG)])

    # 시작점
    if path:
        s = path[0]
        ax.plot(s[1], s[0], "s", color=START_COLOR, ms=_scaled(13, fs),
                mec=MARKER_EDGE, mew=2, zorder=8)
        ax.text(s[1], s[0], "S", ha="center", va="center",
                fontsize=_scaled(12, fs), fontweight="bold",
                color=DARK_BG, zorder=9)

    # 종료점
    if n > 1:
        e = path[-1]
        ax.plot(e[1], e[0], "^", color=UAV_COLOR, ms=_scaled(15, fs),
                mec=MARKER_EDGE, mew=2, zorder=8)

    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(n - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.06)
    cbar.set_label("Step", color=TEXT_CLR, fontsize=_scaled(13, fs))
    cbar.ax.xaxis.set_tick_params(color=TEXT_CLR, labelcolor=TEXT_CLR,
                                  labelsize=_scaled(11, fs))
    cbar.outline.set_edgecolor(BORDER_CLR)

    ax.set_title(title, fontsize=_scaled(18, fs), fontweight="bold", pad=10)
    ax.set_xlabel("Column (Y)", fontsize=_scaled(14, fs))
    ax.set_ylabel("Row (X)", fontsize=_scaled(14, fs))
    ax.tick_params(labelsize=_scaled(11, fs))

    # ── 통계 패널 ──
    ax_info.axis("off")
    moves = n - 1
    energy_used = cum_dist[-1] if cum_dist else 0
    energy_pct = energy_used / env.energy_budget * 100
    n_visited = sum(env.visited)
    mission_ok = all(env.visited)

    status_text = "MISSION COMPLETE" if mission_ok else f"{n_visited}/{env.n_waypoints} WPs"
    status_clr = VISITED_CLR if mission_ok else ACCENT_YELLOW

    stats = [
        ("STATUS",        status_text,                                    status_clr),
        ("Path Steps",    f"{moves}",                                     TEXT_CLR),
        ("Energy Used",   f"{energy_used:.1f} / {env.energy_budget:.1f}", TEXT_CLR),
        ("Energy Spent",  f"{energy_pct:.1f}%",                           TEXT_CLR),
        ("WPs Visited",   f"{n_visited} / {env.n_waypoints}",             TEXT_CLR),
    ]

    y = 0.95
    for label, value, clr in stats:
        ax_info.text(0.08, y, label, transform=ax_info.transAxes,
                     fontsize=_scaled(12, fs), color=MUTED_CLR, va="top")
        ax_info.text(0.08, y - 0.06, value, transform=ax_info.transAxes,
                     fontsize=_scaled(18, fs), fontweight="bold",
                     color=clr, va="top")
        ax_info.text(0.08, y - 0.12, "─" * 22, transform=ax_info.transAxes,
                     fontsize=_scaled(9, fs), color=BORDER_CLR, va="top")
        y -= 0.19

    fig.tight_layout(pad=1.5)
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "best_path.png")
        plt.savefig(save_path, dpi=_PNG_DPI, bbox_inches=None, facecolor=DARK_BG)
    plt.close(fig)


# ── 3a. plot_reward_curve ─────────────────────────────────────────────────────
def plot_reward_curve(rewards, save_path=None, window=50,
                      title="SAC Training Reward Curve"):
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)
    _style_ax(ax)

    eps = np.arange(1, len(rewards) + 1)
    ax.fill_between(eps, rewards, alpha=0.12, color=PATH_START_CLR)
    ax.plot(eps, rewards, alpha=0.2, color=PATH_START_CLR,
            linewidth=0.6, label="Episode Reward")
    ax.axhline(0, color=BORDER_CLR, linewidth=0.7, linestyle="--", alpha=0.6)

    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x_ma = np.arange(window, len(rewards) + 1)
        ax.plot(x_ma, ma, color=PATH_END_CLR, linewidth=2.2,
                label=f"Moving Avg (w={window})")

        best_i = int(np.argmax(ma))
        best_v = ma[best_i]
        ax.annotate(
            f"Best: {best_v:.0f}",
            xy=(x_ma[best_i], best_v),
            xytext=(x_ma[best_i] + len(rewards) * 0.04, best_v),
            fontsize=13, color=ACCENT_YELLOW,
            arrowprops=dict(arrowstyle="->", color=ACCENT_YELLOW, lw=1.2),
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
              fontsize=13)
    ax.grid(True, alpha=0.12, color=GRID_LINE)

    plt.tight_layout()
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "training_curve.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 3b. plot_success_curve ────────────────────────────────────────────────────
def plot_success_curve(success_history, save_path=None, window=50,
                       title="SAC Training Success Rate"):
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)
    _style_ax(ax)

    if len(success_history) >= window:
        sma = np.convolve(success_history, np.ones(window) / window,
                          mode="valid") * 100
        x_sma = np.arange(window, len(success_history) + 1)
        ax.fill_between(x_sma, sma, alpha=0.18, color=VISITED_CLR)
        ax.plot(x_sma, sma, color=VISITED_CLR, linewidth=2.2,
                label=f"Success Rate (w={window})")

    ax.axhline(100, color=BORDER_CLR, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylim(-3, 108)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
              fontsize=13)
    ax.grid(True, alpha=0.12, color=GRID_LINE)

    plt.tight_layout()
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "success_curve.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 3c. plot_actor_loss ──────────────────────────────────────────────────────
def plot_actor_loss(losses, save_path=None, window=50,
                    title="SAC Actor Loss"):
    """Actor(정책) 손실 곡선"""
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)
    _style_ax(ax)

    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, alpha=0.15, color=PATH_START_CLR,
            linewidth=0.6, label="Actor Loss")

    if len(losses) >= window:
        ma = np.convolve(losses, np.ones(window) / window, mode="valid")
        x_ma = np.arange(window, len(losses) + 1)
        ax.plot(x_ma, ma, color="#cf222e", linewidth=2.2,
                label=f"Moving Avg (w={window})")

    ax.set_xlabel("Update Step")
    ax.set_ylabel("Actor Loss")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
              fontsize=13)
    ax.grid(True, alpha=0.12, color=GRID_LINE)

    plt.tight_layout()
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "actor_loss.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 3d. plot_critic_loss ─────────────────────────────────────────────────────
def plot_critic_loss(losses, save_path=None, window=50,
                     title="SAC Critic Loss"):
    """Critic(가치) 손실 곡선"""
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)
    _style_ax(ax)

    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, alpha=0.15, color=PATH_START_CLR,
            linewidth=0.6, label="Critic Loss")

    if len(losses) >= window:
        ma = np.convolve(losses, np.ones(window) / window, mode="valid")
        x_ma = np.arange(window, len(losses) + 1)
        ax.plot(x_ma, ma, color="#8250df", linewidth=2.2,
                label=f"Moving Avg (w={window})")

    ax.set_xlabel("Update Step")
    ax.set_ylabel("Critic Loss (MSE)")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
              fontsize=13)
    ax.grid(True, alpha=0.12, color=GRID_LINE)

    plt.tight_layout()
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "critic_loss.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 3e. plot_alpha_curve (SAC 전용) ───────────────────────────────────────────
def plot_alpha_curve(alphas, save_path=None, window=50,
                     title="SAC Entropy Coefficient (α)"):
    """자동 엔트로피 계수(α) 곡선 — 탐색 정도를 시각화"""
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)
    _style_ax(ax)

    steps = np.arange(1, len(alphas) + 1)
    ax.plot(steps, alphas, alpha=0.15, color=PATH_START_CLR,
            linewidth=0.6, label="α (alpha)")

    if len(alphas) >= window:
        ma = np.convolve(alphas, np.ones(window) / window, mode="valid")
        x_ma = np.arange(window, len(alphas) + 1)
        ax.plot(x_ma, ma, color="#1a7f37", linewidth=2.2,
                label=f"Moving Avg (w={window})")

    ax.set_xlabel("Update Step")
    ax.set_ylabel("α (Entropy Coefficient)")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
              fontsize=13)
    ax.grid(True, alpha=0.12, color=GRID_LINE)

    plt.tight_layout()
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "alpha_curve.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 4. make_flight_gif ────────────────────────────────────────────────────────
def make_flight_gif(env, path=None, save_path="flight.gif",
                    title="UAV Flight (SAC)", fps=5, figsize=None):
    if path is None:
        path = env.path
    if not path:
        print("  [make_flight_gif] path is empty, skipping.")
        return

    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, "flight.gif")

    size = env.grid_size
    fs = _label_scale(size)
    if figsize is None:
        figsize = _gif_figsize(size)

    cmap = _path_cmap()

    bg_img = np.tile(_h2rgb(EMPTY_CLR), (size, size, 1))
    for obs in env.obstacles:
        bg_img[obs[0], obs[1]] = _h2rgb(OBS_COLOR)

    cum_dist = _cum_distances(path)

    def _visited_at(step_i):
        vis = [False] * env.n_waypoints
        for p in path[:step_i + 1]:
            for j, wp in enumerate(env.waypoints):
                if not vis[j]:
                    d = np.sqrt((p[0] - wp[0])**2 + (p[1] - wp[1])**2)
                    if d <= env.config.wp_reach_radius:
                        if j == 0 or all(vis[:j]):
                            vis[j] = True
        return vis

    fig, (ax, ax_info) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": _PANEL_RATIOS}
    )
    _style_fig(fig)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.07, wspace=0.15)

    def draw_frame(step_i):
        ax.cla()
        ax_info.cla()
        _style_ax(ax)
        ax_info.set_facecolor(PANEL_BG)
        for spine in ax_info.spines.values():
            spine.set_edgecolor(BORDER_CLR)

        ax.imshow(bg_img, origin="upper",
                  extent=[-0.5, size - 0.5, size - 0.5, -0.5],
                  aspect="equal", interpolation="nearest", zorder=0)

        _draw_grid_lines(ax, size, zorder=1)

        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)
        ax.set_aspect("equal")

        trail = path[:step_i + 1]
        nt = len(trail)
        if nt > 1:
            for i in range(nt - 1):
                t = i / max(nt - 2, 1)
                ax.plot([trail[i][1], trail[i + 1][1]],
                        [trail[i][0], trail[i + 1][0]],
                        "-", color=cmap(t), linewidth=max(3.0, 2.3 * fs),
                        alpha=0.88, solid_capstyle="round", zorder=3)

        visited = _visited_at(step_i)

        for j, wp in enumerate(env.waypoints):
            clr = VISITED_CLR if visited[j] else WP_COLOR
            ax.plot(wp[1], wp[0], "*", color=clr, ms=_scaled(14, fs),
                    mec=MARKER_EDGE, mew=1.2, zorder=5)
            ax.text(wp[1], wp[0], str(j + 1), ha="center", va="center",
                    fontsize=_scaled(10, fs), fontweight="bold",
                    color="white", zorder=6)

        s = path[0]
        ax.plot(s[1], s[0], "s", color=START_COLOR, ms=_scaled(12, fs),
                mec=MARKER_EDGE, mew=1.5, zorder=5)

        cur = path[step_i]
        ax.plot(cur[1], cur[0], "^", color=UAV_COLOR, ms=_scaled(16, fs),
                mec=MARKER_EDGE, mew=1.8, zorder=7,
                path_effects=[pe.withStroke(linewidth=max(3.0, 2.6 * fs),
                                            foreground=PANEL_BG)])

        energy_left = max(0.0, env.energy_budget - cum_dist[step_i])
        epct = energy_left / env.energy_budget

        ax.set_title(f"{title}  ·  Step {step_i} / {len(path) - 1}",
                     fontsize=_scaled(16, fs), fontweight="bold",
                     pad=8, color=TEXT_CLR)
        ax.set_title(f"{title}  |  Step {step_i} / {len(path) - 1}",
                     fontsize=_scaled(16, fs), fontweight="bold",
                     pad=8, color=TEXT_CLR)
        ax.set_xlabel("Column (Y)", fontsize=_scaled(13, fs))
        ax.set_ylabel("Row (X)", fontsize=_scaled(13, fs))
        ax.tick_params(labelsize=_scaled(10, fs))

        # ── 정보 패널 ──
        ax_info.axis("off")

        bw, bh, bx, by = 0.78, 0.055, 0.11, 0.88
        ax_info.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh, transform=ax_info.transAxes,
            boxstyle="round,pad=0.01", facecolor=GRID_LINE,
            edgecolor=BORDER_CLR, linewidth=0.8, zorder=2))
        bar_clr = (VISITED_CLR if epct > 0.5 else
                   ACCENT_YELLOW if epct > 0.25 else UAV_COLOR)
        if epct > 0:
            ax_info.add_patch(mpatches.FancyBboxPatch(
                (bx, by), bw * epct, bh, transform=ax_info.transAxes,
                boxstyle="round,pad=0.01", facecolor=bar_clr,
                edgecolor="none", zorder=3))
        ax_info.text(0.5, by + bh * 0.5 + 0.003,
                     f"{epct * 100:.0f}%",
                     ha="center", va="center",
                     fontsize=_scaled(14, fs), fontweight="bold",
                     color=TEXT_CLR, transform=ax_info.transAxes, zorder=4)
        ax_info.text(0.5, by + bh + 0.025, "ENERGY",
                     ha="center", va="bottom",
                     fontsize=_scaled(12, fs), color=MUTED_CLR,
                     transform=ax_info.transAxes)

        n_vis = sum(visited)
        done = all(visited)
        stat_rows = [
            ("STEP",   str(step_i),                     TEXT_CLR),
            ("MOVES",  str(step_i),                     TEXT_CLR),
            ("WPs",    f"{n_vis}/{env.n_waypoints}",    VISITED_CLR if done else TEXT_CLR),
            ("STATUS", "DONE!" if done else "In Flight",
             VISITED_CLR if done else MUTED_CLR),
        ]
        y = 0.76
        for lbl, val, clr in stat_rows:
            ax_info.text(0.5, y, lbl, ha="center", va="top",
                         fontsize=_scaled(12, fs),
                         color=MUTED_CLR, transform=ax_info.transAxes)
            ax_info.text(0.5, y - 0.065, val, ha="center", va="top",
                         fontsize=_scaled(18, fs), fontweight="bold", color=clr,
                         transform=ax_info.transAxes)
            ax_info.text(0.5, y - 0.12, "──────────",
                         ha="center", va="top", fontsize=_scaled(9, fs),
                         color=BORDER_CLR, transform=ax_info.transAxes)
            y -= 0.15

        leg_items = [
            mpatches.Patch(facecolor=OBS_COLOR,   label="Obstacle"),
            mpatches.Patch(facecolor=WP_COLOR,    label="Waypoint"),
            mpatches.Patch(facecolor=VISITED_CLR, label="Visited"),
            mpatches.Patch(facecolor=START_COLOR, label="Start"),
            mpatches.Patch(facecolor=UAV_COLOR,   label="UAV"),
        ]
        ax_info.legend(handles=leg_items, loc="lower center",
                       fontsize=_scaled(8, fs), facecolor=DARK_BG,
                       edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
                       bbox_to_anchor=(0.5, 0.01))

    anim = FuncAnimation(fig, draw_frame, frames=len(path),
                          interval=1000 // fps, repeat=False)

    print(f"  Saving GIF: {len(path)} frames @ {fps} fps -> {save_path}")
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=_GIF_DPI,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close(fig)
    print(f"  GIF saved -> {save_path}")


# ── 5. plot_comparison ────────────────────────────────────────────────────────
def plot_comparison(results_dict, window=50,
                     title="Algorithm Comparison", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    _style_fig(fig)
    _style_ax(ax)

    palette = [PATH_END_CLR, PATH_START_CLR, VISITED_CLR,
               ACCENT_YELLOW, "#c084fc", "#fb923c"]

    for idx, (name, rewards) in enumerate(results_dict.items()):
        clr = palette[idx % len(palette)]
        eps = np.arange(1, len(rewards) + 1)
        ax.plot(eps, rewards, alpha=0.15, color=clr, linewidth=0.6)
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window, len(rewards) + 1), ma,
                    color=clr, linewidth=2.2, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Total Reward (Moving Avg, w={window})")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR)
    ax.grid(True, alpha=0.12, color=GRID_LINE)
    ax.axhline(0, color=BORDER_CLR, linewidth=0.7, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
