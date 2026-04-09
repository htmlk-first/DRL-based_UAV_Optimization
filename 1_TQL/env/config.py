"""
UAV Environment Configuration
Grid size, obstacles, energy, waypoints 등을 설정하는 config 클래스
"""
import numpy as np


class EnvConfig:
    def __init__(self, **kwargs):
        # ── Grid ──
        self.grid_size = kwargs.get("grid_size", 20)

        # ── UAV Start Position ──
        self.start_pos = kwargs.get("start_pos", (0, 0))

        # ── Waypoints: (중간 지점, 끝 지점) ──
        default_wp = self._default_waypoints(self.grid_size)
        self.waypoints = kwargs.get("waypoints", default_wp)

        # ── Obstacles ──
        self.obstacle_mode = kwargs.get("obstacle_mode", "fixed")  # "fixed" or "random"
        self.fixed_obstacles = kwargs.get("fixed_obstacles", None)
        self.num_random_obstacles = kwargs.get("num_random_obstacles", 15)
        self.random_seed = kwargs.get("random_seed", 42)

        if self.fixed_obstacles is None:
            self.fixed_obstacles = self._default_obstacles(self.grid_size)

        # ── Energy ──
        self.energy_budget_multiplier = kwargs.get("energy_budget_multiplier", 1.5)
        self.move_cost = kwargs.get("move_cost", 1.0)

        # ── Rewards ──
        self.reward_waypoint = kwargs.get("reward_waypoint", 50.0)
        self.reward_all_done = kwargs.get("reward_all_done", 100.0)
        self.penalty_wall = kwargs.get("penalty_wall", -5.0)
        self.penalty_obstacle = kwargs.get("penalty_obstacle", -10.0)
        self.penalty_step = kwargs.get("penalty_step", -5.0)
        self.penalty_energy_out = kwargs.get("penalty_energy_out", -50.0)

        # ── Max Steps (안전장치) ──
        self.max_steps = kwargs.get("max_steps", self.grid_size * self.grid_size)

    def _default_waypoints(self, size):
        mid = size // 2
        return [(mid, mid), (size - 1, size - 1)]

    def _default_obstacles(self, size):
        """고정 장애물 기본 배치: 그리드 크기에 비례하여 생성"""
        rng = np.random.RandomState(42)
        n = max(10, size * size // 10)
        obstacles = set()
        start = (0, 0)
        mid = (size // 2, size // 2)
        end = (size - 1, size - 1)
        protected = {start, mid, end}

        while len(obstacles) < n:
            pos = (rng.randint(0, size), rng.randint(0, size))
            if pos not in protected:
                obstacles.add(pos)
        return list(obstacles)

    def get_obstacles(self):
        """현재 설정에 따라 장애물 리스트 반환"""
        if self.obstacle_mode == "fixed":
            return list(self.fixed_obstacles)
        else:
            rng = np.random.RandomState(self.random_seed)
            obstacles = set()
            protected = {tuple(self.start_pos)} | {tuple(wp) for wp in self.waypoints}
            n = self.num_random_obstacles
            while len(obstacles) < n:
                pos = (rng.randint(0, self.grid_size), rng.randint(0, self.grid_size))
                if pos not in protected:
                    obstacles.add(pos)
            return list(obstacles)

    def compute_energy_budget(self):
        """최단 경로(맨해튼 거리 합) × multiplier"""
        points = [self.start_pos] + list(self.waypoints)
        total_dist = 0
        for i in range(len(points) - 1):
            total_dist += abs(points[i+1][0] - points[i][0]) + abs(points[i+1][1] - points[i][1])
        return total_dist * self.energy_budget_multiplier
