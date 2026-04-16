"""
UAV Environment Configuration (PPO용)
- 4_1_DDPG와 동일한 환경 설정 유지
- 30×30 공간, 3개 웨이포인트, 연속 좌표 이동
"""
import numpy as np


class EnvConfig:
    def __init__(self, **kwargs):
        # ── Grid ──
        self.grid_size = kwargs.get("grid_size", 30)

        # ── UAV Start Position (셀 중심) ──
        self.start_pos = kwargs.get("start_pos", (0.5, 0.5))

        # ── Waypoints (3개, 셀 중심 좌표) ──
        default_wp = self._default_waypoints(self.grid_size)
        self.waypoints = kwargs.get("waypoints", default_wp)

        # ── Continuous Action ──
        self.max_step_size = kwargs.get("max_step_size", 1.5)
        self.wp_reach_radius = kwargs.get("wp_reach_radius", 1.0)

        # ── Obstacles ──
        self.obstacle_mode = kwargs.get("obstacle_mode", "fixed")
        self.fixed_obstacles = kwargs.get("fixed_obstacles", None)
        self.num_random_obstacles = kwargs.get("num_random_obstacles", 15)
        self.random_seed = kwargs.get("random_seed", 42)

        if self.fixed_obstacles is None:
            self.fixed_obstacles = self._default_obstacles(self.grid_size)

        # ── Energy ──
        self.energy_budget_multiplier = kwargs.get("energy_budget_multiplier", 3.0)

        # ── Rewards ──
        self.reward_waypoint = kwargs.get("reward_waypoint", 100.0)
        self.reward_all_done = kwargs.get("reward_all_done", 300.0)
        self.penalty_wall = kwargs.get("penalty_wall", -5.0)
        self.penalty_obstacle = kwargs.get("penalty_obstacle", -10.0)
        self.penalty_step = kwargs.get("penalty_step", -0.5)
        self.penalty_energy_out = kwargs.get("penalty_energy_out", -50.0)

        # ── Reward Shaping ──
        self.gamma_shaping = kwargs.get("gamma_shaping", 0.99)

        # ── Max Steps ──
        self.max_steps = kwargs.get("max_steps", self.grid_size * 10)

    # ── helpers ──

    def _default_waypoints(self, size):
        """3개 웨이포인트: 1/3 지점, 2/3 지점, 끝 지점 (셀 중심)"""
        third = size // 3
        return [
            (third + 0.5, third + 0.5),
            (third * 2 + 0.5, third * 2 + 0.5),
            (size - 0.5, size - 0.5),
        ]

    def _default_obstacles(self, size):
        rng = np.random.RandomState(42)
        n = max(10, size * size // 10)
        obstacles = set()
        protected = {(0, 0)}
        for wp in self._default_waypoints(size):
            protected.add((int(wp[0]), int(wp[1])))

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
            protected = {(int(self.start_pos[0]), int(self.start_pos[1]))}
            for wp in self.waypoints:
                protected.add((int(wp[0]), int(wp[1])))
            n = self.num_random_obstacles
            while len(obstacles) < n:
                pos = (rng.randint(0, self.grid_size), rng.randint(0, self.grid_size))
                if pos not in protected:
                    obstacles.add(pos)
            return list(obstacles)

    def compute_energy_budget(self):
        """웨이포인트 간 유클리드 거리 합 × 배율"""
        points = [self.start_pos] + list(self.waypoints)
        total_dist = 0
        for i in range(len(points) - 1):
            total_dist += np.sqrt(
                (points[i + 1][0] - points[i][0]) ** 2
                + (points[i + 1][1] - points[i][1]) ** 2
            )
        return total_dist * self.energy_budget_multiplier
