"""
UAV 3D Continuous-Action Environment (DDPG 3D용)
- Gymnasium 호환 인터페이스
- 3_2_DDQN_3D 기반 → DDPG 확장:
  · 연속 행동 공간 → Box(-1,1, shape=(3,))  (dx, dy, dz)
  · 연속 좌표 이동 (실수 좌표, 임의 방향/크기)
  · 14방향 장애물 감지 (3_2_DDQN_3D: 6방향)
  · 유클리드 거리 기반 에너지 소비 + z축 추가 비용
  · 건물형 3D 장애물
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig3D


class UAVEnv3D(gym.Env):
    """
    3D 연속 좌표 기반 UAV 경로 최적화 + 에너지 제약 환경

    Actions: Box(-1, 1, shape=(3,))  → (dx, dy, dz) scaled by max_step_size
    State:   [x/sx, y/sy, z/sz, energy/budget,
              wp_visited...(n_waypoints),
              adj_14방향,
              dx_next_wp/sx, dy_next_wp/sy, dz_next_wp/sz]
    """
    metadata = {"render_modes": ["human"]}

    # 14방향 오프셋: 6 면 + 8 꼭짓점 대각선
    ADJ_OFFSETS = [
        # 6 face directions
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        # 8 vertex diagonals
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else EnvConfig3D()
        self.grid_size_x = self.config.grid_size_x
        self.grid_size_y = self.config.grid_size_y
        self.grid_size_z = self.config.grid_size_z
        self.waypoints = [tuple(wp) for wp in self.config.waypoints]
        self.n_waypoints = len(self.waypoints)
        self.obstacles = set(map(tuple, self.config.get_obstacles()))
        self.buildings = self.config.get_buildings()
        self.energy_budget = self.config.compute_energy_budget()

        # ── Continuous action space (3D) ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

        # 상태 차원: pos(3) + energy(1) + wp_visited(n) + adj_14 + dir_wp(3)
        obs_dim = 3 + 1 + self.n_waypoints + 14 + 3
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.uav_pos = None
        self.energy = None
        self.visited = None
        self.steps = None
        self.path = None

    # ── Gymnasium API ──

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav_pos = list(self.config.start_pos)
        self.energy = self.energy_budget
        self.visited = [False] * self.n_waypoints
        self.steps = 0
        self.path = [tuple(self.uav_pos)]
        self._prev_potential = self._potential()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.steps += 1

        dx = float(action[0]) * self.config.max_step_size
        dy = float(action[1]) * self.config.max_step_size
        dz = float(action[2]) * self.config.max_step_size_z
        nx = self.uav_pos[0] + dx
        ny = self.uav_pos[1] + dy
        nz = self.uav_pos[2] + dz

        # 에너지 비용: XY 유클리드 + Z축 추가 비용
        xy_dist = np.sqrt(dx * dx + dy * dy)
        z_dist = abs(dz)
        move_cost = xy_dist + z_dist * self.config.z_cost_multiplier

        reward = self.config.penalty_step
        terminated = False
        truncated = False
        info = {"event": "move"}

        # ── 경계 충돌 ──
        if (nx < 0 or nx >= self.grid_size_x or
            ny < 0 or ny >= self.grid_size_y or
            nz < 0 or nz >= self.grid_size_z):
            reward += self.config.penalty_wall
            info["event"] = "wall_collision"

        # ── 장애물(건물) 충돌 (목적지 셀 기준) ──
        elif (int(nx), int(ny), int(nz)) in self.obstacles:
            reward += self.config.penalty_obstacle
            info["event"] = "obstacle_collision"

        # ── 정상 이동 ──
        else:
            self.uav_pos = [nx, ny, nz]
            self.energy -= move_cost
            self.path.append(tuple(self.uav_pos))

            # 웨이포인트 도달 체크 (3D 유클리드 거리 기반)
            for i, wp in enumerate(self.waypoints):
                if not self.visited[i]:
                    dist_wp = np.sqrt(
                        (self.uav_pos[0] - wp[0]) ** 2
                        + (self.uav_pos[1] - wp[1]) ** 2
                        + (self.uav_pos[2] - wp[2]) ** 2
                    )
                    if dist_wp <= self.config.wp_reach_radius:
                        if i == 0 or all(self.visited[:i]):
                            self.visited[i] = True
                            reward += self.config.reward_waypoint
                            info["event"] = f"waypoint_{i}_reached"
                    break

            if all(self.visited):
                reward += self.config.reward_all_done
                terminated = True
                info["event"] = "mission_complete"

        # ── 에너지 소진 ──
        if self.energy <= 0:
            reward += self.config.penalty_energy_out
            terminated = True
            info["event"] = "energy_depleted"

        # ── 최대 스텝 초과 ──
        if self.steps >= self.config.max_steps:
            truncated = True
            info["event"] = "max_steps_reached"

        info["energy"] = self.energy
        info["visited"] = list(self.visited)
        info["steps"] = self.steps

        # ── Potential-based reward shaping ──
        new_potential = self._potential()
        shaping = self.config.gamma_shaping * new_potential - self._prev_potential
        reward += shaping
        self._prev_potential = new_potential

        return self._get_obs(), reward, terminated, truncated, info

    # ── internal helpers ──

    def _get_obs(self):
        """정규화된 상태 벡터 반환"""
        sx = max(self.grid_size_x - 1, 1)
        sy = max(self.grid_size_y - 1, 1)
        sz = max(self.grid_size_z - 1, 1)
        x, y, z = self.uav_pos
        cell_x, cell_y, cell_z = int(x), int(y), int(z)

        # 14방향 장애물/경계 감지 (6 면 + 8 꼭짓점 대각선)
        adj = []
        for ox, oy, oz in self.ADJ_OFFSETS:
            cx = cell_x + ox
            cy = cell_y + oy
            cz = cell_z + oz
            if (cx < 0 or cx >= self.grid_size_x or
                cy < 0 or cy >= self.grid_size_y or
                cz < 0 or cz >= self.grid_size_z):
                adj.append(1.0)
            elif (cx, cy, cz) in self.obstacles:
                adj.append(1.0)
            else:
                adj.append(0.0)

        # 다음 목표 WP까지의 정규화 방향
        target = self._next_target()
        dx_wp = (target[0] - x) / sx
        dy_wp = (target[1] - y) / sy
        dz_wp = (target[2] - z) / sz

        obs = np.array(
            [x / sx, y / sy, z / sz,
             self.energy / max(self.energy_budget, 1e-8)]
            + [float(v) for v in self.visited]
            + adj
            + [dx_wp, dy_wp, dz_wp],
            dtype=np.float32,
        )
        return obs

    def _next_target(self):
        """다음 미방문 WP 좌표 반환"""
        for i, wp in enumerate(self.waypoints):
            if not self.visited[i]:
                return wp
        return self.waypoints[-1]

    def _potential(self):
        """potential-based reward shaping: 다음 WP까지 3D 유클리드 거리의 음수"""
        target = self._next_target()
        return -np.sqrt(
            (self.uav_pos[0] - target[0]) ** 2
            + (self.uav_pos[1] - target[1]) ** 2
            + (self.uav_pos[2] - target[2]) ** 2
        )

    # ── 시각화 보조 ──

    def render(self, mode="human"):
        x, y, z = self.uav_pos
        print(f"\nStep: {self.steps} | Energy: {self.energy:.1f} | "
              f"Pos: ({x:.1f},{y:.1f},{z:.1f}) | Visited: {self.visited}")
        print(f"Grid: {self.grid_size_x}x{self.grid_size_y}x{self.grid_size_z} | "
              f"Buildings: {len(self.buildings)} ({len(self.obstacles)} cells) | "
              f"Path len: {len(self.path)}")

        if self.grid_size_x * self.grid_size_y > 2500:
            print("Layer render skipped for large grids; use saved PNG/GIF for visualization.")
            return

        iz = int(z)
        symbols = {0: "· ", 1: "██", 2: "◎ ", 3: "✈ ", 4: "✓ "}
        print(f"\n── Z={iz} layer ──")
        print("+" + "--" * self.grid_size_y + "+")
        for xi in range(self.grid_size_x):
            line = "|"
            for yi in range(self.grid_size_y):
                if int(x) == xi and int(y) == yi:
                    line += symbols[3]
                elif (xi, yi, iz) in self.obstacles:
                    line += symbols[1]
                elif any(
                    int(wp[0]) == xi and int(wp[1]) == yi
                    and int(wp[2]) == iz and self.visited[j]
                    for j, wp in enumerate(self.waypoints)
                ):
                    line += symbols[4]
                elif any(
                    int(wp[0]) == xi and int(wp[1]) == yi
                    and int(wp[2]) == iz and not self.visited[j]
                    for j, wp in enumerate(self.waypoints)
                ):
                    line += symbols[2]
                else:
                    line += symbols[0]
            line += "|"
            print(line)
        print("+" + "--" * self.grid_size_y + "+")
