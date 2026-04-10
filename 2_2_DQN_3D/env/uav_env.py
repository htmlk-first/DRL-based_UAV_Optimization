"""
UAV 3D Grid Environment (Discrete Action Space)
- Gymnasium 호환 인터페이스
- DQN 3D용: 정규화된 연속 상태 벡터 반환
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig3D


class UAVEnv3D(gym.Env):
    """
    3D 그리드 기반 UAV 경로 최적화 + 에너지 제약 환경

    Actions: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
    State: [x/sx, y/sy, z/sz, energy/budget, wp_visited...,
            adj_6방향 (0=free, 1=blocked),
            dx_next_wp/sx, dy_next_wp/sy, dz_next_wp/sz]
    """
    metadata = {"render_modes": ["human"]}

    ACTION_XP = 0  # +X
    ACTION_XN = 1  # -X
    ACTION_YP = 2  # +Y
    ACTION_YN = 3  # -Y
    ACTION_ZP = 4  # +Z (상승)
    ACTION_ZN = 5  # -Z (하강)
    ACTION_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    MOVES = {
        0: (1, 0, 0), 1: (-1, 0, 0),
        2: (0, 1, 0), 3: (0, -1, 0),
        4: (0, 0, 1), 5: (0, 0, -1),
    }

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else EnvConfig3D()
        self.grid_size_x = self.config.grid_size_x
        self.grid_size_y = self.config.grid_size_y
        self.grid_size_z = self.config.grid_size_z
        self.waypoints = [tuple(wp) for wp in self.config.waypoints]
        self.n_waypoints = len(self.waypoints)
        self.obstacles = set(map(tuple, self.config.get_obstacles()))
        self.buildings = self.config.get_buildings()  # {(x,y): height}
        self.energy_budget = self.config.compute_energy_budget()

        self.action_space = spaces.Discrete(6)

        # State: [x, y, z, energy, wp_visited..., adj_6, dx_wp, dy_wp, dz_wp]
        obs_dim = 3 + 1 + self.n_waypoints + 6 + 3
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
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.steps += 1

        dx, dy, dz = self.MOVES[action]
        nx = self.uav_pos[0] + dx
        ny = self.uav_pos[1] + dy
        nz = self.uav_pos[2] + dz

        reward = self.config.penalty_step
        terminated = False
        truncated = False
        info = {"event": "move"}

        # 벽 충돌 검사
        if (nx < 0 or nx >= self.grid_size_x or
            ny < 0 or ny >= self.grid_size_y or
            nz < 0 or nz >= self.grid_size_z):
            reward += self.config.penalty_wall
            info["event"] = "wall_collision"
        elif (nx, ny, nz) in self.obstacles:
            reward += self.config.penalty_obstacle
            info["event"] = "obstacle_collision"
        else:
            self.uav_pos = [nx, ny, nz]
            # 수직 이동은 에너지 소모 다름
            cost = self.config.move_cost_z if dz != 0 else self.config.move_cost
            self.energy -= cost
            self.path.append(tuple(self.uav_pos))

            for i, wp in enumerate(self.waypoints):
                if not self.visited[i] and tuple(self.uav_pos) == wp:
                    if i == 0 or all(self.visited[:i]):
                        self.visited[i] = True
                        reward += self.config.reward_waypoint
                        info["event"] = f"waypoint_{i}_reached"
                    break

            if all(self.visited):
                reward += self.config.reward_all_done
                terminated = True
                info["event"] = "mission_complete"

        if self.energy <= 0:
            reward += self.config.penalty_energy_out
            terminated = True
            info["event"] = "energy_depleted"

        if self.steps >= self.config.max_steps:
            truncated = True
            info["event"] = "max_steps_reached"

        info["energy"] = self.energy
        info["visited"] = list(self.visited)
        info["steps"] = self.steps

        # Potential-based reward shaping
        new_potential = self._potential()
        shaping = self.config.gamma_shaping * new_potential - self._prev_potential
        reward += shaping
        self._prev_potential = new_potential

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """정규화된 상태 벡터"""
        sx = max(self.grid_size_x - 1, 1)
        sy = max(self.grid_size_y - 1, 1)
        sz = max(self.grid_size_z - 1, 1)
        x, y, z = self.uav_pos

        # 6방향 인접 셀 장애물 여부
        adj = []
        for ddx, ddy, ddz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = x + ddx, y + ddy, z + ddz
            if (nx < 0 or nx >= self.grid_size_x or
                ny < 0 or ny >= self.grid_size_y or
                nz < 0 or nz >= self.grid_size_z):
                adj.append(1.0)
            elif (nx, ny, nz) in self.obstacles:
                adj.append(1.0)
            else:
                adj.append(0.0)

        # 다음 목표 WP 방향 (정규화)
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
        for i, wp in enumerate(self.waypoints):
            if not self.visited[i]:
                return wp
        return self.waypoints[-1]

    def _potential(self):
        target = self._next_target()
        return -(
            abs(self.uav_pos[0] - target[0])
            + abs(self.uav_pos[1] - target[1])
            + abs(self.uav_pos[2] - target[2])
        )

    def render(self, mode="human"):
        x, y, z = self.uav_pos
        print(f"\nStep: {self.steps} | Energy: {self.energy:.1f} | "
              f"Pos: ({x},{y},{z}) | Visited: {self.visited}")
        print(f"Grid: {self.grid_size_x}x{self.grid_size_y}x{self.grid_size_z} | "
              f"Buildings: {len(self.buildings)} ({len(self.obstacles)} cells) | "
              f"Path len: {len(self.path)}")

        # XY 슬라이스 출력 (현재 z 레이어)
        symbols = {0: "· ", 1: "██", 2: "◎ ", 3: "✈ ", 4: "✓ "}
        print(f"\n── Z={z} layer ──")
        print("+" + "--" * self.grid_size_y + "+")
        for xi in range(self.grid_size_x):
            line = "|"
            for yi in range(self.grid_size_y):
                if [xi, yi, z] == list(self.uav_pos):
                    line += symbols[3]
                elif (xi, yi, z) in self.obstacles:
                    line += symbols[1]
                elif any(
                    tuple(wp) == (xi, yi, z) and self.visited[j]
                    for j, wp in enumerate(self.waypoints)
                ):
                    line += symbols[4]
                elif any(
                    tuple(wp) == (xi, yi, z) and not self.visited[j]
                    for j, wp in enumerate(self.waypoints)
                ):
                    line += symbols[2]
                else:
                    line += symbols[0]
            line += "|"
            print(line)
        print("+" + "--" * self.grid_size_y + "+")
