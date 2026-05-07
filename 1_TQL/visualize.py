"""
UAV 환경 시각화 모듈 (라이트 테마)
- plot_grid           : 현재 환경 상태
- plot_path           : 비행 경로 + 통계 패널
- plot_training_curve : 학습 곡선 (reward + success rate)
- plot_qvalue_heatmap : Q-value 히트맵 (TQL 전용)
- make_flight_gif     : 비행 경로 GIF 애니메이션
- plot_comparison     : 다중 알고리즘 비교
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter


# ── 공통 색상 팔레트 ──────────────────────────────────────────────────────────
DARK_BG    = "#ffffff"   # 흰색 배경
PANEL_BG   = "#f6f8fa"   # 연한 회색 패널
GRID_LINE  = "#d0d7de"   # 격자선
BORDER_CLR = "#afb8c1"   # 테두리

UAV_COLOR      = "#cf222e"   # 진빨강 — UAV
PATH_START_CLR = "#0969da"   # 진파랑 — 경로 시작
PATH_END_CLR   = "#cf222e"   # 진빨강 — 경로 끝
WP_COLOR       = "#e5534b"   # 산호   — 미방문 WP
VISITED_CLR    = "#1a7f37"   # 진초록 — 방문 WP
OBS_COLOR      = "#57606a"   # 중간회 — 장애물
START_COLOR    = "#bf8700"   # 진황금 — 시작점
EMPTY_CLR      = "#f6f8fa"   # 연회색 — 빈 셀
TEXT_CLR       = "#24292f"
MUTED_CLR      = "#57606a"
ACCENT_YELLOW  = "#9a6700"

MARKER_EDGE    = "#555555"   # 마커 테두리 (라이트 배경용)

# ── 통일된 출력 규격 ─────────────────────────────────────────────────────────
# Best Path / Flight GIF 모두 16:9, 동일한 패널 비율로 고정
_FIG_SIZE      = (16, 9)
_PANEL_RATIOS  = [1.4, 1.0]
_PNG_DPI       = 140
_GIF_DPI       = 100


def _h2rgb(hex_color: str):
    """hex → [r, g, b] float 0-1"""
    h = hex_color.lstrip("#")
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
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
    """셀 배경 + 장애물을 numpy 이미지로 렌더링"""
    size = env.grid_size
    img = np.tile(_h2rgb(EMPTY_CLR), (size, size, 1))
    for obs in env.obstacles:
        img[obs[0], obs[1]] = _h2rgb(OBS_COLOR)

    ax.imshow(img, origin="upper",
              extent=[-0.5, size - 0.5, size - 0.5, -0.5],
              aspect="equal", interpolation="nearest", zorder=0)

    for i in range(size + 1):
        ax.axhline(i - 0.5, color=GRID_LINE, linewidth=0.4, alpha=0.6, zorder=1)
        ax.axvline(i - 0.5, color=GRID_LINE, linewidth=0.4, alpha=0.6, zorder=1)


# ── 1. plot_grid ──────────────────────────────────────────────────────────────
def plot_grid(env, title="UAV Grid Environment", save_path=None):
    """현재 환경 상태 시각화 (다크 테마)"""
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
def plot_path(env, path=None, title="UAV Flight Path", save_path=None):
    """비행 경로 시각화 (그라디언트 경로 + 통계 패널)"""
    if path is None:
        path = env.path
    size = env.grid_size
    cmap = _path_cmap()
    n = len(path)

    fig, (ax, ax_info) = plt.subplots(
        1, 2, figsize=_FIG_SIZE,
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

    # 경로 그라디언트
    if n > 1:
        for i in range(n - 1):
            t = i / max(n - 2, 1)
            ax.plot([path[i][1], path[i + 1][1]],
                    [path[i][0], path[i + 1][0]],
                    "-", color=cmap(t), linewidth=2.8, alpha=0.9,
                    solid_capstyle="round", zorder=3)
        for i, p in enumerate(path):
            t = i / max(n - 1, 1)
            ax.plot(p[1], p[0], "o", color=cmap(t), ms=3.5, alpha=0.55, zorder=4)

    # 에너지 추적
    # env.path는 성공적 이동만 포함 → step_i == 이동 횟수
    energy_at_wp = {}
    for step_i, pos in enumerate(path):
        e_left = env.energy_budget - step_i * env.config.move_cost
        for j, wp in enumerate(env.waypoints):
            if tuple(pos) == wp and j not in energy_at_wp:
                energy_at_wp[j] = e_left

    # 웨이포인트
    for i, wp in enumerate(env.waypoints):
        clr = VISITED_CLR if env.visited[i] else WP_COLOR
        ax.plot(wp[1], wp[0], "*", color=clr, ms=18,
                mec=MARKER_EDGE, mew=1, zorder=6)
        ax.text(wp[1], wp[0], str(i + 1), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=7)
        ax.text(wp[1] + 0.38, wp[0] - 0.42, f"WP{i + 1}",
                fontsize=13, fontweight="bold", color=clr, zorder=7,
                path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)])
        if i in energy_at_wp:
            pct = energy_at_wp[i] / env.energy_budget * 100
            ax.text(wp[1] + 0.38, wp[0] + 0.42,
                    f"E:{pct:.0f}%", fontsize=11, color=VISITED_CLR, zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)])

    # 시작점
    if path:
        s = path[0]
        ax.plot(s[1], s[0], "s", color=START_COLOR, ms=13,
                mec=MARKER_EDGE, mew=2, zorder=8)
        ax.text(s[1], s[0], "S", ha="center", va="center",
                fontsize=13, fontweight="bold", color=DARK_BG, zorder=9)

    # 종료점
    if n > 1:
        e = path[-1]
        ax.plot(e[1], e[0], "^", color=UAV_COLOR, ms=14,
                mec=MARKER_EDGE, mew=2, zorder=8)

    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(n - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.06)
    cbar.set_label("Step", color=TEXT_CLR, fontsize=13)
    cbar.ax.xaxis.set_tick_params(color=TEXT_CLR, labelcolor=TEXT_CLR, labelsize=7)
    cbar.outline.set_edgecolor(BORDER_CLR)

    ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
    ax.set_xlabel("Column (Y)")
    ax.set_ylabel("Row (X)")

    # ── 통계 패널 ──
    ax_info.axis("off")
    moves = n - 1
    energy_used = moves * env.config.move_cost
    energy_pct = energy_used / env.energy_budget * 100
    n_visited = sum(env.visited)
    mission_ok = all(env.visited)

    status_text = "MISSION COMPLETE" if mission_ok else f"{n_visited}/{env.n_waypoints} WPs"
    status_clr = VISITED_CLR if mission_ok else ACCENT_YELLOW

    stats = [
        ("STATUS",        status_text,                                    status_clr),
        ("Path Steps",    f"{moves}",                                     TEXT_CLR),
        ("Energy Used",   f"{energy_used:.0f} / {env.energy_budget:.0f}", TEXT_CLR),
        ("Energy Spent",  f"{energy_pct:.1f}%",                           TEXT_CLR),
        ("WPs Visited",   f"{n_visited} / {env.n_waypoints}",             TEXT_CLR),
    ]

    y = 0.95
    for label, value, clr in stats:
        ax_info.text(0.08, y, label, transform=ax_info.transAxes,
                     fontsize=11, color=MUTED_CLR, va="top")
        ax_info.text(0.08, y - 0.055, value, transform=ax_info.transAxes,
                     fontsize=17, fontweight="bold", color=clr, va="top")
        ax_info.text(0.08, y - 0.10, "─" * 22, transform=ax_info.transAxes,
                     fontsize=10, color=BORDER_CLR, va="top")
        y -= 0.17

    fig.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=_PNG_DPI, bbox_inches=None, facecolor=DARK_BG)
    plt.close(fig)


# ── 3a. plot_reward_curve ─────────────────────────────────────────────────────
def plot_reward_curve(rewards, window=50,
                      title="Training Reward Curve", save_path=None):
    """학습 곡선: 에피소드별 Total Reward + 이동평균"""
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


# ── 3b. plot_success_curve ────────────────────────────────────────────────────
def plot_success_curve(success_history, window=50,
                       title="Training Success Rate", save_path=None):
    """학습 곡선: 에피소드별 성공률 이동평균"""
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


# ── 4. plot_qvalue_heatmap ────────────────────────────────────────────────────
def plot_qvalue_heatmap(agent, env, waypoint_idx=0,
                         title="Q-Value Heatmap", save_path=None):
    """
    TQL Q-value 히트맵
    waypoint_idx : 몇 번째 WP까지 방문한 상태를 시각화할지 (0 = 아무것도 방문 안 함)

    NOTE: 모든 WP 방문 완료 상태(waypoint_idx == n_waypoints)는 에피소드 종료
          즉시 터미널 상태이므로 Q-테이블에 저장되지 않아 히트맵이 비어있음.
    """
    size = env.grid_size
    visited_state = tuple(i < waypoint_idx for i in range(env.n_waypoints))

    q_map = np.full((size, size), np.nan)
    act_map = np.full((size, size), -1, dtype=int)

    # mid_energy 단일값 대신 해당 visited_state의 모든 에너지 레벨을 순회해
    # 셀별 최대 Q값(및 그 때의 행동)을 선택
    for key, q_vals in agent.q_table.items():
        r, c, _, vis = key
        if vis != visited_state:
            continue
        if (r, c) in env.obstacles:
            continue
        cell_max = float(np.max(q_vals))
        if np.isnan(q_map[r, c]) or cell_max > q_map[r, c]:
            q_map[r, c] = cell_max
            act_map[r, c] = int(np.argmax(q_vals))

    fig, ax = plt.subplots(figsize=(max(7, size * 0.6), max(7, size * 0.6)))
    _style_fig(fig)
    _style_ax(ax)

    masked = np.ma.masked_where(np.isnan(q_map), q_map)
    im = ax.imshow(masked, cmap="plasma", origin="upper",
                   extent=[-0.5, size - 0.5, size - 0.5, -0.5],
                   aspect="equal", interpolation="nearest", zorder=1)

    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle(
            (obs[1] - 0.5, obs[0] - 0.5), 1, 1,
            facecolor="#d0d7de", edgecolor=GRID_LINE, linewidth=0.5,
            alpha=0.92, zorder=2))

    for i in range(size + 1):
        ax.axhline(i - 0.5, color=GRID_LINE, linewidth=0.35, alpha=0.5, zorder=3)
        ax.axvline(i - 0.5, color=GRID_LINE, linewidth=0.35, alpha=0.5, zorder=3)

    # 행동 화살표
    arrow_delta = {0: (0, -0.32), 1: (0, 0.32), 2: (-0.32, 0), 3: (0.32, 0)}
    for r in range(size):
        for c in range(size):
            if act_map[r, c] >= 0:
                dx, dy = arrow_delta[act_map[r, c]]
                ax.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                            arrowprops=dict(arrowstyle="->", color=TEXT_CLR,
                                            lw=0.7, alpha=0.65),
                            zorder=4)

    for i, wp in enumerate(env.waypoints):
        clr = VISITED_CLR if i < waypoint_idx else WP_COLOR
        ax.plot(wp[1], wp[0], "*", color=clr, ms=14,
                mec=MARKER_EDGE, mew=1, zorder=6)
        ax.text(wp[1], wp[0], str(i + 1), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=7)

    s = env.config.start_pos
    ax.plot(s[1], s[0], "s", color=START_COLOR, ms=11,
            mec=MARKER_EDGE, mew=1.5, zorder=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Max Q-Value", color=TEXT_CLR, fontsize=14)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR, labelcolor=TEXT_CLR, labelsize=7)
    cbar.outline.set_edgecolor(BORDER_CLR)

    visited_str = ("None" if waypoint_idx == 0 else
                   " → ".join(f"WP{i + 1}" for i in range(waypoint_idx)))
    ax.set_title(f"{title}\nVisited: {visited_str}  |  Max Q across all energy levels",
                 fontsize=17, fontweight="bold", pad=8)
    ax.set_xlabel("Column (Y)")
    ax.set_ylabel("Row (X)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ── 5. make_flight_gif ────────────────────────────────────────────────────────
def make_flight_gif(env, path=None, title="UAV Flight",
                    save_path="flight.gif", fps=5, figsize=None):
    """
    UAV 비행 경로 GIF 애니메이션 생성

    Parameters
    ----------
    env       : UAVEnv 인스턴스 (장애물·웨이포인트 정보 필요)
    path      : [(row, col), ...] — None 이면 env.path 사용
    title     : GIF 제목
    save_path : 저장 경로 (.gif)
    fps       : 초당 프레임 수
    figsize   : (w, h) — None 이면 자동
    """
    if path is None:
        path = env.path
    if not path:
        print("  [make_flight_gif] path is empty, skipping.")
        return

    size = env.grid_size
    if figsize is None:
        figsize = _FIG_SIZE

    cmap = _path_cmap()

    # 배경 이미지 사전 계산 (장애물 포함)
    bg_img = np.tile(_h2rgb(EMPTY_CLR), (size, size, 1))
    for obs in env.obstacles:
        bg_img[obs[0], obs[1]] = _h2rgb(OBS_COLOR)

    def _visited_at(step_i):
        """step_i 번째 위치까지의 WP 방문 여부 재구성"""
        vis = [False] * env.n_waypoints
        for p in path[:step_i + 1]:
            for j, wp in enumerate(env.waypoints):
                if not vis[j] and tuple(p) == wp:
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

        # 배경
        ax.imshow(bg_img, origin="upper",
                  extent=[-0.5, size - 0.5, size - 0.5, -0.5],
                  aspect="equal", interpolation="nearest", zorder=0)

        for i in range(size + 1):
            ax.axhline(i - 0.5, color=GRID_LINE, lw=0.35, alpha=0.55, zorder=1)
            ax.axvline(i - 0.5, color=GRID_LINE, lw=0.35, alpha=0.55, zorder=1)

        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)
        ax.set_aspect("equal")

        # 경로 궤적 (그라디언트)
        trail = path[:step_i + 1]
        nt = len(trail)
        if nt > 1:
            for i in range(nt - 1):
                t = i / max(nt - 2, 1)
                ax.plot([trail[i][1], trail[i + 1][1]],
                        [trail[i][0], trail[i + 1][0]],
                        "-", color=cmap(t), linewidth=2.8,
                        alpha=0.88, solid_capstyle="round", zorder=3)

        visited = _visited_at(step_i)

        # 웨이포인트
        for j, wp in enumerate(env.waypoints):
            clr = VISITED_CLR if visited[j] else WP_COLOR
            ax.plot(wp[1], wp[0], "*", color=clr, ms=14,
                    mec=MARKER_EDGE, mew=1, zorder=5)
            ax.text(wp[1], wp[0], str(j + 1), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white", zorder=6)

        # 시작점
        s = path[0]
        ax.plot(s[1], s[0], "s", color=START_COLOR, ms=12,
                mec=MARKER_EDGE, mew=1.5, zorder=5)

        # UAV (현재 위치)
        cur = path[step_i]
        ax.plot(cur[1], cur[0], "^", color=UAV_COLOR, ms=16,
                mec=MARKER_EDGE, mew=1.8, zorder=7,
                path_effects=[pe.withStroke(linewidth=3, foreground=PANEL_BG)])

        # env.path는 성공 이동만 포함 → step_i == 이동 횟수
        moves = step_i
        energy_left = max(0.0, env.energy_budget - moves * env.config.move_cost)
        epct = energy_left / env.energy_budget

        ax.set_title(f"{title}  ·  Step {step_i} / {len(path) - 1}",
                     fontsize=15, fontweight="bold", pad=7, color=TEXT_CLR)
        ax.set_xlabel("Column (Y)", fontsize=13)
        ax.set_ylabel("Row (X)", fontsize=13)

        # ── 정보 패널 ──
        ax_info.axis("off")

        # 에너지 바
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
                     ha="center", va="bottom", fontsize=11, color=MUTED_CLR,
                     transform=ax_info.transAxes)

        # 통계
        n_vis = sum(visited)
        done = all(visited)
        stat_rows = [
            ("STEP",   str(step_i),                     TEXT_CLR),
            ("MOVES",  str(moves),                      TEXT_CLR),
            ("WPs",    f"{n_vis}/{env.n_waypoints}",    VISITED_CLR if done else TEXT_CLR),
            ("STATUS", "DONE!" if done else "In Flight",
             VISITED_CLR if done else MUTED_CLR),
        ]
        y = 0.76
        for lbl, val, clr in stat_rows:
            ax_info.text(0.5, y, lbl, ha="center", va="top", fontsize=11,
                         color=MUTED_CLR, transform=ax_info.transAxes)
            ax_info.text(0.5, y - 0.057, val, ha="center", va="top",
                         fontsize=17, fontweight="bold", color=clr,
                         transform=ax_info.transAxes)
            ax_info.text(0.5, y - 0.098, "──────────",
                         ha="center", va="top", fontsize=10,
                         color=BORDER_CLR, transform=ax_info.transAxes)
            y -= 0.155

        leg_items = [
            mpatches.Patch(facecolor=OBS_COLOR,   label="Obstacle"),
            mpatches.Patch(facecolor=WP_COLOR,    label="Waypoint"),
            mpatches.Patch(facecolor=VISITED_CLR, label="Visited"),
            mpatches.Patch(facecolor=START_COLOR, label="Start"),
            mpatches.Patch(facecolor=UAV_COLOR,   label="UAV"),
        ]
        ax_info.legend(handles=leg_items, loc="lower center",
                       fontsize=11, facecolor=DARK_BG,
                       edgecolor=BORDER_CLR, labelcolor=TEXT_CLR,
                       bbox_to_anchor=(0.5, 0.01))

    anim = FuncAnimation(fig, draw_frame, frames=len(path),
                          interval=1000 // fps, repeat=False)

    print(f"  Saving GIF: {len(path)} frames @ {fps} fps → {save_path}")
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=_GIF_DPI,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close(fig)
    print(f"  GIF saved → {save_path}")


# ── 6. plot_comparison ────────────────────────────────────────────────────────
def plot_comparison(results_dict, window=50,
                     title="Algorithm Comparison", save_path=None):
    """여러 알고리즘의 학습 곡선 비교"""
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
