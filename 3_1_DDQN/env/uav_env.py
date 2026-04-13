"""
UAV Grid Environment (Discrete Action Space)
- Gymnasium 호환 인터페이스
- DDQN용: 정규화된 연속 상태 벡터 반환
- 2_1_DQN 대비 확장: 3개 웨이포인트 지원
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig


class UAVEnv(gym.Env):
    """
    2D 그리드 기반 UAV 경로 최적화 + 에너지 제약 환경

    Actions: 0=상, 1=하, 2=좌, 3=우
    State: [x/size, y/size, energy/budget, wp_visited...,
            adj_up, adj_down, adj_left, adj_right,
            dx_next_wp/size, dy_next_wp/size]
    """
    metadata = {"render_modes": ["human"]}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else EnvConfig()
        self.grid_size = self.config.grid_size
        self.waypoints = [tuple(wp) for wp in self.config.waypoints]
        self.n_waypoints = len(self.waypoints)
        self.obstacles = set(map(tuple, self.config.get_obstacles()))
        self.energy_budget = self.config.compute_energy_budget()

        self.action_space = spaces.Discrete(4)

        # 정규화된 상태:
        # [x, y, energy, wp_visited...(n_waypoints), adj_4방향, dx_wp, dy_wp]
        obs_dim = 2 + 1 + self.n_waypoints + 4 + 2
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

        dx, dy = self.MOVES[action]
        nx, ny = self.uav_pos[0] + dx, self.uav_pos[1] + dy

        reward = self.config.penalty_step
        terminated = False
        truncated = False
        info = {"event": "move"}

        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
            reward += self.config.penalty_wall
            info["event"] = "wall_collision"
        elif (nx, ny) in self.obstacles:
            reward += self.config.penalty_obstacle
            info["event"] = "obstacle_collision"
        else:
            self.uav_pos = [nx, ny]
            self.energy -= self.config.move_cost
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
        """정규화된 상태 벡터 반환"""
        size = max(self.grid_size - 1, 1)
        x, y = self.uav_pos

        # 인접 셀 장애물 여부
        adj = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                adj.append(1.0)
            elif (nx, ny) in self.obstacles:
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
        """potential-based reward shaping: 다음 WP까지 맨해튼 거리의 음수"""
        target = self._next_target()
        return -(abs(self.uav_pos[0] - target[0])
                 + abs(self.uav_pos[1] - target[1]))

    def get_grid_map(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 1
        for i, wp in enumerate(self.waypoints):
            if self.visited[i]:
                grid[wp[0], wp[1]] = 4
            else:
                grid[wp[0], wp[1]] = 2
        grid[self.uav_pos[0], self.uav_pos[1]] = 3
        return grid

    def render(self, mode="human"):
        grid = self.get_grid_map()
        symbols = {0: "· ", 1: "██", 2: "◎ ", 3: "✈ ", 4: "✓ "}
        print(f"\nStep: {self.steps} | Energy: {self.energy:.1f} | Visited: {self.visited}")
        print("+" + "--" * self.grid_size + "+")
        for row in grid:
            line = "|" + "".join(symbols.get(c, "? ") for c in row) + "|"
            print(line)
        print("+" + "--" * self.grid_size + "+")
