"""
UAV Environment Configuration (DDQN용)
- 2_1_DQN 대비 확장: 30×30 그리드, 3개 웨이포인트
"""
import numpy as np


class EnvConfig:
    def __init__(self, **kwargs):
        # ── Grid ──
        self.grid_size = kwargs.get("grid_size", 50)

        # ── UAV Start Position ──
        self.start_pos = kwargs.get("start_pos", (0, 0))

        # ── Waypoints (3개로 확장) ──
        default_wp = self._default_waypoints(self.grid_size)
        self.waypoints = kwargs.get("waypoints", default_wp)

        # ── Obstacles ──
        self.obstacle_mode = kwargs.get("obstacle_mode", "fixed")
        self.fixed_obstacles = kwargs.get("fixed_obstacles", None)
        default_random_obstacles = max(25, self.grid_size * self.grid_size // 60)
        self.num_random_obstacles = kwargs.get(
            "num_random_obstacles", default_random_obstacles
        )
        self.random_seed = kwargs.get("random_seed", 42)

        if self.fixed_obstacles is None:
            self.fixed_obstacles = self._default_obstacles(self.grid_size)

        # ── Energy ──
        self.energy_budget_multiplier = kwargs.get("energy_budget_multiplier", 2.8)
        self.move_cost = kwargs.get("move_cost", 1.0)

        # ── Rewards ──
        self.reward_waypoint = kwargs.get("reward_waypoint", 100.0)
        self.reward_all_done = kwargs.get("reward_all_done", 300.0)
        self.penalty_wall = kwargs.get("penalty_wall", -5.0)
        self.penalty_obstacle = kwargs.get("penalty_obstacle", -10.0)
        self.penalty_step = kwargs.get("penalty_step", -1.0)
        self.penalty_energy_out = kwargs.get("penalty_energy_out", -50.0)

        # ── Reward Shaping ──
        self.gamma_shaping = kwargs.get("gamma_shaping", 0.99)

        # ── Max Steps ──
        self.max_steps = kwargs.get("max_steps", self.grid_size * 30)

    def _default_waypoints(self, size):
        """3개 웨이포인트: 1/3 지점, 2/3 지점, 끝 지점"""
        third = size // 3
        return [
            (third, third),
            (third * 2, third * 2),
            (size - 1, size - 1),
        ]

    def _default_obstacles(self, size):
        rng = np.random.RandomState(42)
        n = max(10, size * size // 10)
        obstacles = set()
        # 시작점 + 웨이포인트 보호
        protected = {(0, 0)}
        for wp in self._default_waypoints(size):
            protected.add(wp)

        while len(obstacles) < n:
            pos = (rng.randint(0, size), rng.randint(0, size))
            if pos not in protected:
                obstacles.add(pos)
        return list(obstacles)

    def get_obstacles(self):
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
        points = [self.start_pos] + list(self.waypoints)
        total_dist = 0
        for i in range(len(points) - 1):
            total_dist += (abs(points[i + 1][0] - points[i][0])
                           + abs(points[i + 1][1] - points[i][1]))
        return total_dist * self.energy_budget_multiplier
