"""
UAV 3D Environment Configuration (PPO 3D용)
- 4_2_DDPG_3D 기반 → PPO 3D 확장
- 30×30×5 3D 공간, 3개 웨이포인트, 건물형 장애물
- 연속 좌표 이동 (실수 좌표, 임의 방향/크기)
"""
import numpy as np


class EnvConfig3D:
    def __init__(self, **kwargs):
        # ── Grid ──
        self.grid_size_x = kwargs.get("grid_size_x", 30)
        self.grid_size_y = kwargs.get("grid_size_y", 30)
        self.grid_size_z = kwargs.get("grid_size_z", 5)

        # ── UAV Start Position (셀 중심 좌표) ──
        self.start_pos = kwargs.get("start_pos", (0.5, 0.5, 0.5))

        # ── Waypoints (3개, 셀 중심 좌표) ──
        default_wp = self._default_waypoints()
        self.waypoints = kwargs.get("waypoints", default_wp)

        # ── Continuous Action ──
        self.max_step_size = kwargs.get("max_step_size", 1.5)
        self.max_step_size_z = kwargs.get("max_step_size_z", 1.0)
        self.wp_reach_radius = kwargs.get("wp_reach_radius", 1.0)

        # ── Obstacles (건물형: 지상 z=0에서 랜덤 높이) ──
        self.obstacle_mode = kwargs.get("obstacle_mode", "fixed")
        self.fixed_obstacles = kwargs.get("fixed_obstacles", None)
        self.num_buildings = kwargs.get("num_buildings", 40)
        self.min_building_height = kwargs.get("min_building_height", 1)
        self.max_building_height = kwargs.get("max_building_height", None)
        self.random_seed = kwargs.get("random_seed", 42)

        self._buildings = {}
        if self.fixed_obstacles is None:
            self.fixed_obstacles = self._default_obstacles()

        # ── Energy ──
        self.energy_budget_multiplier = kwargs.get("energy_budget_multiplier", 3.0)
        self.z_cost_multiplier = kwargs.get("z_cost_multiplier", 2.5)

        # ── Rewards ──
        self.reward_waypoint = kwargs.get("reward_waypoint", 100.0)
        self.reward_all_done = kwargs.get("reward_all_done", 300.0)
        self.penalty_wall = kwargs.get("penalty_wall", -5.0)
        self.penalty_obstacle = kwargs.get("penalty_obstacle", -10.0)
        self.penalty_step = kwargs.get("penalty_step", -0.5)
        self.penalty_energy_out = kwargs.get("penalty_energy_out", -50.0)
        self.penalty_z_reversal = kwargs.get("penalty_z_reversal", -1.0)

        # ── Reward Shaping ──
        self.gamma_shaping = kwargs.get("gamma_shaping", 0.99)

        # ── Max Steps ──
        self.max_steps = kwargs.get(
            "max_steps",
            self.grid_size_x * 10,
        )

    # ── helpers ──

    def _default_waypoints(self):
        """3개 웨이포인트: 1/3, 2/3, 끝 지점 (셀 중심, z축 변화 포함)"""
        third_x = self.grid_size_x // 3
        third_y = self.grid_size_y // 3
        mz = self.grid_size_z // 2
        return [
            (third_x + 0.5, third_y + 0.5, mz + 0.5),
            (third_x * 2 + 0.5, third_y * 2 + 0.5, mz + 0.5),
            (self.grid_size_x - 0.5, self.grid_size_y - 0.5,
             self.grid_size_z - 0.5),
        ]

    def _default_obstacles(self):
        """건물형 장애물 생성: (x,y) 위치에 z=0부터 랜덤 높이까지"""
        rng = np.random.RandomState(self.random_seed)
        max_h = self.max_building_height or (self.grid_size_z - 1)
        min_h = self.min_building_height

        protected_xy = set()
        for pos3d in [self.start_pos] + list(self._default_waypoints()):
            protected_xy.add((int(pos3d[0]), int(pos3d[1])))
        protected_3d = {
            (int(self.start_pos[0]), int(self.start_pos[1]),
             int(self.start_pos[2]))
        }
        for wp in self._default_waypoints():
            protected_3d.add((int(wp[0]), int(wp[1]), int(wp[2])))

        buildings = {}
        while len(buildings) < self.num_buildings:
            bx = rng.randint(0, self.grid_size_x)
            by = rng.randint(0, self.grid_size_y)
            if (bx, by) in protected_xy or (bx, by) in buildings:
                continue
            height = rng.randint(min_h, max_h + 1)
            buildings[(bx, by)] = height

        obstacles = set()
        for (bx, by), h in buildings.items():
            for bz in range(h):
                cell = (bx, by, bz)
                if cell not in protected_3d:
                    obstacles.add(cell)

        self._buildings = buildings
        return list(obstacles)

    def get_obstacles(self):
        if self.obstacle_mode == "fixed":
            return list(self.fixed_obstacles)
        rng = np.random.RandomState(self.random_seed)
        max_h = self.max_building_height or (self.grid_size_z - 1)
        min_h = self.min_building_height

        protected_xy = set()
        for pos3d in [self.start_pos] + list(self.waypoints):
            protected_xy.add((int(pos3d[0]), int(pos3d[1])))
        protected_3d = {
            (int(self.start_pos[0]), int(self.start_pos[1]),
             int(self.start_pos[2]))
        }
        for wp in self.waypoints:
            protected_3d.add((int(wp[0]), int(wp[1]), int(wp[2])))

        buildings = {}
        while len(buildings) < self.num_buildings:
            bx = rng.randint(0, self.grid_size_x)
            by = rng.randint(0, self.grid_size_y)
            if (bx, by) in protected_xy or (bx, by) in buildings:
                continue
            height = rng.randint(min_h, max_h + 1)
            buildings[(bx, by)] = height

        obstacles = set()
        for (bx, by), h in buildings.items():
            for bz in range(h):
                cell = (bx, by, bz)
                if cell not in protected_3d:
                    obstacles.add(cell)

        self._buildings = buildings
        return list(obstacles)

    def get_buildings(self):
        if not self._buildings:
            self.get_obstacles()
        return dict(self._buildings)

    def compute_energy_budget(self):
        """웨이포인트 간 유클리드 거리 합 × 배율 (z축 추가 비용 반영)"""
        points = [self.start_pos] + list(self.waypoints)
        total_cost = 0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            dz = points[i + 1][2] - points[i][2]
            xy_dist = np.sqrt(dx ** 2 + dy ** 2)
            z_dist = abs(dz) * self.z_cost_multiplier
            total_cost += xy_dist + z_dist
        return total_cost * self.energy_budget_multiplier
