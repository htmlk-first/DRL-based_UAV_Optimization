"""
UAV Continuous-Action Environment (DDPG용)
- Gymnasium 호환 인터페이스
- DDQN 대비 확장:
  · 연속 행동 공간 → Box(-1,1, shape=(2,))  (dx, dy)
  · 연속 좌표 이동 (실수 좌표, 임의 방향/크기)
  · 8방향 장애물 감지 (DDQN: 4방향)
  · 유클리드 거리 기반 에너지 소비
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig


class UAVEnv(gym.Env):
    """
    2D 연속 좌표 기반 UAV 경로 최적화 + 에너지 제약 환경

    Actions: Box(-1, 1, shape=(2,))  → (dx, dy) scaled by max_step_size
    State:   [x/size, y/size, energy/budget,
              wp_visited...(n_waypoints),
              adj_8방향,
              dx_next_wp/size, dy_next_wp/size]
    """
    metadata = {"render_modes": ["human"]}

    # 8방향 오프셋 (4 cardinal + 4 diagonal)
    ADJ_OFFSETS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else EnvConfig()
        self.grid_size = self.config.grid_size
        self.waypoints = [tuple(wp) for wp in self.config.waypoints]
        self.n_waypoints = len(self.waypoints)
        self.obstacles = set(map(tuple, self.config.get_obstacles()))
        self.energy_budget = self.config.compute_energy_budget()

        # ── Continuous action space ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        # 상태 차원: pos(2) + energy(1) + wp_visited(n) + adj_8(8) + dir_wp(2)
        obs_dim = 2 + 1 + self.n_waypoints + 8 + 2
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
        nx = self.uav_pos[0] + dx
        ny = self.uav_pos[1] + dy
        move_dist = np.sqrt(dx * dx + dy * dy)

        reward = self.config.penalty_step
        terminated = False
        truncated = False
        info = {"event": "move"}

        # ── 경계 충돌 ──
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
            reward += self.config.penalty_wall
            info["event"] = "wall_collision"

        # ── 장애물 충돌 (목적지 셀 기준) ──
        elif (int(nx), int(ny)) in self.obstacles:
            reward += self.config.penalty_obstacle
            info["event"] = "obstacle_collision"

        # ── 정상 이동 ──
        else:
            self.uav_pos = [nx, ny]
            self.energy -= move_dist
            self.path.append(tuple(self.uav_pos))

            # 웨이포인트 도달 체크 (유클리드 거리 기반)
            for i, wp in enumerate(self.waypoints):
                if not self.visited[i]:
                    dist_wp = np.sqrt(
                        (self.uav_pos[0] - wp[0]) ** 2
                        + (self.uav_pos[1] - wp[1]) ** 2
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
        size = max(self.grid_size - 1, 1)
        x, y = self.uav_pos
        cell_x, cell_y = int(x), int(y)

        # 8방향 장애물/경계 감지
        adj = []
        for ox, oy in self.ADJ_OFFSETS:
            cx, cy = cell_x + ox, cell_y + oy
            if cx < 0 or cx >= self.grid_size or cy < 0 or cy >= self.grid_size:
                adj.append(1.0)
            elif (cx, cy) in self.obstacles:
                adj.append(1.0)
            else:
                adj.append(0.0)

        # 다음 목표 WP까지의 정규화 방향
        target = self._next_target()
        dx_wp = (target[0] - x) / size
        dy_wp = (target[1] - y) / size

        obs = np.array(
            [x / size, y / size,
             self.energy / max(self.energy_budget, 1e-8)]
            + [float(v) for v in self.visited]
            + adj
            + [dx_wp, dy_wp],
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
        """potential-based reward shaping: 다음 WP까지 유클리드 거리의 음수"""
        target = self._next_target()
        return -np.sqrt(
            (self.uav_pos[0] - target[0]) ** 2
            + (self.uav_pos[1] - target[1]) ** 2
        )

    # ── 시각화 보조 ──

    def get_grid_map(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 1
        for i, wp in enumerate(self.waypoints):
            wr, wc = int(wp[0]), int(wp[1])
            if self.visited[i]:
                grid[wr, wc] = 4
            else:
                grid[wr, wc] = 2
        ur, uc = int(self.uav_pos[0]), int(self.uav_pos[1])
        ur = min(ur, self.grid_size - 1)
        uc = min(uc, self.grid_size - 1)
        grid[ur, uc] = 3
        return grid

    def render(self, mode="human"):
        grid = self.get_grid_map()
        symbols = {0: "· ", 1: "██", 2: "◎ ", 3: "✈ ", 4: "✓ "}
        print(f"\nStep: {self.steps} | Energy: {self.energy:.1f}"
              f" | Visited: {self.visited}")
        print("+" + "--" * self.grid_size + "+")
        for row in grid:
            line = "|" + "".join(symbols.get(c, "? ") for c in row) + "|"
            print(line)
        print("+" + "--" * self.grid_size + "+")
