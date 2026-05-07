"""
UAV 3D 환경 시각화 모듈 (라이트 테마, PPO 3D용)
- plot_path_3d         : 3D 비행 경로 + 통계 패널
- plot_reward_curve    : 학습 보상 곡선
- plot_success_curve   : 성공률 곡선
- plot_policy_loss     : Policy 손실 곡선 (PPO)
- plot_value_loss      : Value 손실 곡선 (PPO)
- plot_entropy_curve   : 엔트로피 곡선 (PPO)
- make_flight_gif_3d   : 3D 비행 경로 GIF 애니메이션
- plot_comparison      : 다중 알고리즘 비교
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

# ── 통일된 출력 규격 ─────────────────────────────────────────────────────────
_FIG_SIZE      = (16, 9)
_PANEL_RATIOS  = [1.4, 1.0]
_PNG_DPI       = 140
_GIF_DPI       = 100

# Fallback for legacy building metadata. New environments provide size_x/size_y
# so rendered footprint matches actual obstacle cells.
BUILDING_FOOTPRINT_SIZE = 1.0


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


def _style_ax3d(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.zaxis.label.set_color(TEXT_CLR)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_LINE)
    ax.yaxis.pane.set_edgecolor(GRID_LINE)
    ax.zaxis.pane.set_edgecolor(GRID_LINE)
    ax.grid(True, alpha=0.15, color=GRID_LINE)


def _style_fig(fig):
    fig.patch.set_facecolor(DARK_BG)


def _path_cmap():
    return LinearSegmentedColormap.from_list("uav_trail",
                                              [PATH_START_CLR, PATH_END_CLR])


def _cum_distances_3d(path):
    """경로를 따라 누적 3D 유클리드 거리 계산"""
    dists = [0.0]
    for i in range(len(path) - 1):
        d = np.sqrt((path[i+1][0] - path[i][0])**2
                    + (path[i+1][1] - path[i][1])**2
                    + (path[i+1][2] - path[i][2])**2)
        dists.append(dists[-1] + d)
    return dists


def _draw_building(ax, x, y, height, color, alpha=0.3,
                   footprint_x=None, footprint_y=None):
    if footprint_x is None:
        footprint_x = BUILDING_FOOTPRINT_SIZE
    if footprint_y is None:
        footprint_y = footprint_x
    sx = footprint_x / 2
    sy = footprint_y / 2
    z_lo = -0.4
    z_hi = height - 0.6
    if z_hi <= z_lo:
        z_hi = z_lo + 0.2
    verts = np.array([
        [x-sx, y-sy, z_lo], [x+sx, y-sy, z_lo], [x+sx, y+sy, z_lo], [x-sx, y+sy, z_lo],
        [x-sx, y-sy, z_hi], [x+sx, y-sy, z_hi], [x+sx, y+sy, z_hi], [x-sx, y+sy, z_hi],
    ])
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[0], verts[3], verts[7], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
    ]
    face_colors = [color] * 6
    roof_rgb = [min(1.0, c + 0.15) for c in color]
    face_colors[1] = roof_rgb

    ax.add_collection3d(Poly3DCollection(
        faces, alpha=alpha, facecolor=face_colors,
        edgecolor=(*color, 0.6), linewidth=0.4))


def _draw_cube(ax, x, y, z, color, alpha=0.3, size=0.8):
    s = size / 2
    verts = np.array([
        [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y+s, z-s], [x-s, y+s, z-s],
        [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s],
    ])
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[0], verts[3], verts[7], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
    ]
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=alpha, facecolor=color,
        edgecolor=(*color, 0.5), linewidth=0.3))


def _building_meta(center, data):
    cx, cy = center
    if isinstance(data, dict):
        height = int(data.get("height", 1))
        size_x = int(data.get("size_x", data.get("footprint_x", 1)))
        size_y = int(data.get("size_y", data.get("footprint_y", size_x)))
        x0 = int(data.get("x0", round(cx - (size_x - 1) / 2.0)))
        y0 = int(data.get("y0", round(cy - (size_y - 1) / 2.0)))
    else:
        height = int(data)
        size_x = int(round(BUILDING_FOOTPRINT_SIZE))
        size_y = size_x
        x0 = int(round(cx - (size_x - 1) / 2.0))
        y0 = int(round(cy - (size_y - 1) / 2.0))
    return height, size_x, size_y, x0, y0


def _draw_obstacle_cubes(ax, obstacles, alpha=0.25, buildings=None):
    obs_rgb = _h2rgb(OBS_COLOR)
    if buildings:
        drawn = set()
        for (bx, by), data in buildings.items():
            h, size_x, size_y, x0, y0 = _building_meta((bx, by), data)
            _draw_building(ax, bx, by, h, color=obs_rgb, alpha=alpha,
                           footprint_x=size_x, footprint_y=size_y)
            for ix in range(x0, x0 + size_x):
                for iy in range(y0, y0 + size_y):
                    for bz in range(h):
                        drawn.add((ix, iy, bz))
        for ox, oy, oz in obstacles:
            if (ox, oy, oz) not in drawn:
                _draw_cube(ax, ox, oy, oz, color=obs_rgb, alpha=alpha)
    else:
        for ox, oy, oz in obstacles:
            _draw_cube(ax, ox, oy, oz, color=obs_rgb, alpha=alpha)


def _draw_3d_scene(ax, env, trail=None, visited=None):
    sx = env.grid_size_x
    sy = env.grid_size_y
    sz = env.grid_size_z

    ax.set_xlim(-0.5, sx - 0.5)
    ax.set_ylim(-0.5, sy - 0.5)
    ax.set_zlim(-0.5, sz - 0.5)
    ax.set_xlabel("X", fontsize=18)
    ax.set_ylabel("Y", fontsize=18)
    ax.set_zlabel("Z", fontsize=18)

    buildings = getattr(env, 'buildings', None)
    _draw_obstacle_cubes(ax, env.obstacles, alpha=0.18, buildings=buildings)

    vis = visited if visited is not None else env.visited
    for i, wp in enumerate(env.waypoints):
        clr = VISITED_CLR if vis[i] else WP_COLOR
        ax.scatter(*wp, s=200, color=clr, marker="*",
                   edgecolors=MARKER_EDGE, linewidth=0.8, zorder=5,
                   depthshade=False)
        ax.text(wp[0] + 0.3, wp[1] + 0.3, wp[2] + 0.3,
                f"WP{i+1}", fontsize=15, color=clr, fontweight="bold")

    s = env.config.start_pos
    ax.scatter(*s, s=120, color=START_COLOR, marker="s",
               edgecolors=MARKER_EDGE, linewidth=1, zorder=5,
               depthshade=False)

    if trail and len(trail) > 1:
        cmap = _path_cmap()
        n = len(trail)
        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]
        zs = [p[2] for p in trail]
        for i in range(n - 1):
            t = i / max(n - 2, 1)
            ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                    "-", color=cmap(t), linewidth=2.5, alpha=0.85, zorder=3)


# ── 1. plot_path_3d ───────────────────────────────────────────────────────────
def plot_path_3d(env, path=None, title="UAV 3D Flight Path", save_path=None):
    if path is None:
        path = env.path
    n = len(path)

    fig = plt.figure(figsize=_FIG_SIZE)
    _style_fig(fig)

    gs = fig.add_gridspec(1, 2, width_ratios=_PANEL_RATIOS,
                          left=0.04, right=0.97, top=0.95, bottom=0.06,
                          wspace=0.12)
    ax = fig.add_subplot(gs[0], projection="3d")
    _style_ax3d(ax)

    _draw_3d_scene(ax, env, trail=path)

    if path:
        last = path[-1]
        ax.scatter(*last, s=180, color=UAV_COLOR, marker="^",
                   edgecolors=MARKER_EDGE, linewidth=1.2, zorder=7,
                   depthshade=False)

    cmap = _path_cmap()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(n - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
    cbar.set_label("Step", color=TEXT_CLR, fontsize=17)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR, labelcolor=TEXT_CLR, labelsize=9)
    cbar.outline.set_edgecolor(BORDER_CLR)

    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.view_init(elev=25, azim=-60)

    # ── 통계 패널 ──
    ax_info = fig.add_subplot(gs[1])
    ax_info.set_facecolor(PANEL_BG)
    for spine in ax_info.spines.values():
        spine.set_edgecolor(BORDER_CLR)
    ax_info.axis("off")

    moves = n - 1
    # 에너지 사용 (XY 유클리드 + Z 추가 비용)
    energy_used = 0
    for i in range(1, n):
        dxy = np.sqrt((path[i][0] - path[i-1][0])**2
                      + (path[i][1] - path[i-1][1])**2)
        dz = abs(path[i][2] - path[i-1][2])
        energy_used += dxy + dz * env.config.z_cost_multiplier
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
        ("Grid Size",     f"{env.grid_size_x}×{env.grid_size_y}×{env.grid_size_z}", MUTED_CLR),
        ("Buildings",     f"{len(getattr(env, 'buildings', {}))} ({len(env.obstacles)} cells)", MUTED_CLR),
    ]

    y = 0.92
    for label, value, clr in stats:
        ax_info.text(0.08, y, label, transform=ax_info.transAxes,
                     fontsize=18, color=MUTED_CLR, va="top")
        ax_info.text(0.08, y - 0.05, value, transform=ax_info.transAxes,
                     fontsize=17, fontweight="bold", color=clr, va="top")
        ax_info.text(0.08, y - 0.095, "─" * 22, transform=ax_info.transAxes,
                     fontsize=14, color=BORDER_CLR, va="top")
        y -= 0.13

    if save_path:
        plt.savefig(save_path, dpi=_PNG_DPI, bbox_inches=None, facecolor=DARK_BG)
    plt.close(fig)


# ── 2. plot_reward_curve ──────────────────────────────────────────────────────
def plot_reward_curve(rewards, window=50,
                      title="Training Reward Curve", save_path=None):
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
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 3. plot_success_curve ─────────────────────────────────────────────────────
def plot_success_curve(success_history, window=50,
                       title="Training Success Rate", save_path=None):
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
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 4. plot_policy_loss (PPO) ─────────────────────────────────────────────────
def plot_policy_loss(losses, window=50,
                     title="PPO 3D Policy Loss", save_path=None):
    """PPO 정책 손실 곡선"""
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)
    _style_ax(ax)

    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, alpha=0.15, color=PATH_START_CLR,
            linewidth=0.6, label="Update Loss")

    if len(losses) >= window:
        ma = np.convolve(losses, np.ones(window) / window, mode="valid")
        x_ma = np.arange(window, len(losses) + 1)
        ax.plot(x_ma, ma, color="#cf222e", linewidth=2.2,
                label=f"Moving Avg (w={window})")

    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Policy Loss")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
              fontsize=13)
    ax.grid(True, alpha=0.12, color=GRID_LINE)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 5. plot_value_loss (PPO) ──────────────────────────────────────────────────
def plot_value_loss(losses, window=50,
                    title="PPO 3D Value Loss", save_path=None):
    """PPO 가치 손실 곡선"""
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)
    _style_ax(ax)

    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, alpha=0.15, color=PATH_START_CLR,
            linewidth=0.6, label="Update Loss")

    if len(losses) >= window:
        ma = np.convolve(losses, np.ones(window) / window, mode="valid")
        x_ma = np.arange(window, len(losses) + 1)
        ax.plot(x_ma, ma, color="#8250df", linewidth=2.2,
                label=f"Moving Avg (w={window})")

    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Value Loss (MSE)")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
              fontsize=13)
    ax.grid(True, alpha=0.12, color=GRID_LINE)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 6. plot_entropy_curve (PPO) ───────────────────────────────────────────────
def plot_entropy_curve(entropies, window=50,
                       title="PPO 3D Entropy", save_path=None):
    """정책 엔트로피 곡선 — 탐색 정도를 시각화"""
    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)
    _style_ax(ax)

    steps = np.arange(1, len(entropies) + 1)
    ax.plot(steps, entropies, alpha=0.15, color=PATH_START_CLR,
            linewidth=0.6, label="Entropy")

    if len(entropies) >= window:
        ma = np.convolve(entropies, np.ones(window) / window, mode="valid")
        x_ma = np.arange(window, len(entropies) + 1)
        ax.plot(x_ma, ma, color="#1a7f37", linewidth=2.2,
                label=f"Moving Avg (w={window})")

    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Entropy")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
              fontsize=13)
    ax.grid(True, alpha=0.12, color=GRID_LINE)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 7. make_flight_gif_3d ────────────────────────────────────────────────────
def make_flight_gif_3d(env, path=None, title="UAV 3D Flight",
                       save_path="flight_3d.gif", fps=5):
    if path is None:
        path = env.path
    if not path:
        print("  [make_flight_gif_3d] path is empty, skipping.")
        return

    def _visited_at(step_i):
        vis = [False] * env.n_waypoints
        for p in path[:step_i + 1]:
            for j, wp in enumerate(env.waypoints):
                if not vis[j]:
                    d = np.sqrt((p[0] - wp[0])**2
                                + (p[1] - wp[1])**2
                                + (p[2] - wp[2])**2)
                    if d <= env.config.wp_reach_radius:
                        if j == 0 or all(vis[:j]):
                            vis[j] = True
        return vis

    fig = plt.figure(figsize=_FIG_SIZE)
    _style_fig(fig)

    gs = fig.add_gridspec(1, 2, width_ratios=_PANEL_RATIOS,
                          left=0.04, right=0.97, top=0.93, bottom=0.07,
                          wspace=0.15)
    ax = fig.add_subplot(gs[0], projection="3d")
    ax_info = fig.add_subplot(gs[1])

    def draw_frame(step_i):
        ax.cla()
        ax_info.cla()
        _style_ax3d(ax)
        ax_info.set_facecolor(PANEL_BG)
        for spine in ax_info.spines.values():
            spine.set_edgecolor(BORDER_CLR)

        trail = path[:step_i + 1]
        visited = _visited_at(step_i)

        _draw_3d_scene(ax, env, trail=trail, visited=visited)

        cur = path[step_i]
        ax.scatter(*cur, s=200, color=UAV_COLOR, marker="^",
                   edgecolors=MARKER_EDGE, linewidth=1.5, zorder=7,
                   depthshade=False)

        ax.set_title(f"{title}  ·  Step {step_i} / {len(path) - 1}",
                     fontsize=16, fontweight="bold", pad=8, color=TEXT_CLR)
        ax.view_init(elev=25, azim=-60 + step_i * 0.5)

        # ── 정보 패널 ──
        ax_info.axis("off")

        energy_used = 0
        for i in range(1, min(step_i + 1, len(path))):
            dxy = np.sqrt((path[i][0] - path[i-1][0])**2
                          + (path[i][1] - path[i-1][1])**2)
            dz = abs(path[i][2] - path[i-1][2])
            energy_used += dxy + dz * env.config.z_cost_multiplier
        energy_left = max(0.0, env.energy_budget - energy_used)
        epct = energy_left / env.energy_budget

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
                     ha="center", va="center", fontsize=14, fontweight="bold",
                     color=TEXT_CLR, transform=ax_info.transAxes, zorder=4)
        ax_info.text(0.5, by + bh + 0.025, "ENERGY",
                     ha="center", va="bottom", fontsize=18, color=MUTED_CLR,
                     transform=ax_info.transAxes)

        n_vis = sum(visited)
        done = all(visited)
        stat_rows = [
            ("STEP",     str(step_i),                                  TEXT_CLR),
            ("POSITION", f"({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f})",  TEXT_CLR),
            ("WPs",      f"{n_vis}/{env.n_waypoints}",
             VISITED_CLR if done else TEXT_CLR),
            ("STATUS",   "DONE!" if done else "In Flight",
             VISITED_CLR if done else MUTED_CLR),
        ]
        y = 0.76
        for lbl, val, clr in stat_rows:
            ax_info.text(0.5, y, lbl, ha="center", va="top", fontsize=18,
                         color=MUTED_CLR, transform=ax_info.transAxes)
            ax_info.text(0.5, y - 0.065, val, ha="center", va="top",
                         fontsize=18, fontweight="bold", color=clr,
                         transform=ax_info.transAxes)
            ax_info.text(0.5, y - 0.12, "──────────",
                         ha="center", va="top", fontsize=14,
                         color=BORDER_CLR, transform=ax_info.transAxes)
            y -= 0.18

        leg_items = [
            mpatches.Patch(facecolor=OBS_COLOR,   label="Obstacle"),
            mpatches.Patch(facecolor=WP_COLOR,    label="Waypoint"),
            mpatches.Patch(facecolor=VISITED_CLR, label="Visited"),
            mpatches.Patch(facecolor=START_COLOR, label="Start"),
            mpatches.Patch(facecolor=UAV_COLOR,   label="UAV"),
        ]
        ax_info.legend(handles=leg_items, loc="lower center",
                       fontsize=17, facecolor=DARK_BG,
                       edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
                       bbox_to_anchor=(0.5, 0.01))

    anim = FuncAnimation(fig, draw_frame, frames=len(path),
                          interval=1000 // fps, repeat=False)

    print(f"  Saving 3D GIF: {len(path)} frames @ {fps} fps -> {save_path}")
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=_GIF_DPI,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close(fig)
    print(f"  GIF saved -> {save_path}")


# ── 8. plot_comparison ────────────────────────────────────────────────────────
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

    ax.set_xlabel("Episode", fontsize=20)
    ax.set_ylabel(f"Total Reward (Moving Avg, w={window})", fontsize=20)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_CLR, labelcolor=TEXT_CLR)
    ax.grid(True, alpha=0.12, color=GRID_LINE)
    ax.axhline(0, color=BORDER_CLR, linewidth=0.7, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
