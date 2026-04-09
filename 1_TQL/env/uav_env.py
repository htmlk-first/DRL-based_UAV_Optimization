"""
UAV Grid Environment (Discrete Action Space)
- Gymnasium 호환 인터페이스
- TQL, DQN, DDQN 등 이산 행동 기반 알고리즘 공용
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig


class UAVEnv(gym.Env):
    """
    2D 그리드 기반 UAV 경로 최적화 + 에너지 제약 환경

    Actions: 0=상, 1=하, 2=좌, 3=우
    State: [uav_x, uav_y, energy_remaining, wp1_visited, wp2_visited]
    """
    metadata = {"render_modes": ["human"]}

    # 행동 매핑: 상, 하, 좌, 우
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

        # Gym spaces
        self.action_space = spaces.Discrete(4)

        # 상태: [x, y, energy, wp1_visited, wp2_visited, ...]
        low = np.zeros(2 + 1 + self.n_waypoints, dtype=np.float32)
        high = np.array(
            [self.grid_size - 1, self.grid_size - 1, self.energy_budget]
            + [1.0] * self.n_waypoints,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 내부 상태
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

        # 벽 충돌 체크
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
            reward += self.config.penalty_wall
            info["event"] = "wall_collision"
        # 장애물 충돌 체크
        elif (nx, ny) in self.obstacles:
            reward += self.config.penalty_obstacle
            info["event"] = "obstacle_collision"
        else:
            # 이동 성공
            self.uav_pos = [nx, ny]
            self.energy -= self.config.move_cost
            self.path.append(tuple(self.uav_pos))

            # 웨이포인트 체크 (순서대로 방문해야 함)
            for i, wp in enumerate(self.waypoints):
                if not self.visited[i] and tuple(self.uav_pos) == wp:
                    # 이전 웨이포인트가 모두 방문되었는지 확인
                    if i == 0 or all(self.visited[:i]):
                        self.visited[i] = True
                        reward += self.config.reward_waypoint
                        info["event"] = f"waypoint_{i}_reached"
                    break

            # 모든 웨이포인트 방문 완료
            if all(self.visited):
                reward += self.config.reward_all_done
                terminated = True
                info["event"] = "mission_complete"

        # 에너지 소진 체크
        if self.energy <= 0:
            reward += self.config.penalty_energy_out
            terminated = True
            info["event"] = "energy_depleted"

        # 최대 스텝 초과
        if self.steps >= self.config.max_steps:
            truncated = True
            info["event"] = "max_steps_reached"

        info["energy"] = self.energy
        info["visited"] = list(self.visited)
        info["steps"] = self.steps

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.array(
            self.uav_pos + [self.energy] + [float(v) for v in self.visited],
            dtype=np.float32,
        )
        return obs

    def get_state_tuple(self):
        """TQL용: 이산 상태를 tuple로 반환 (Q-테이블 키로 사용)"""
        energy_level = int(self.energy)
        return (self.uav_pos[0], self.uav_pos[1], energy_level, tuple(self.visited))

    def get_grid_map(self):
        """시각화용: 현재 그리드 맵 반환"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # 0: empty, 1: obstacle, 2: waypoint, 3: UAV, 4: visited_waypoint
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
