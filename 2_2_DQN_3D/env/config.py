"""
UAV 3D Environment Configuration (DQN 3D용)
"""
import numpy as np


class EnvConfig3D:
    def __init__(self, **kwargs):
        # ── Grid ──
        self.grid_size_x = kwargs.get("grid_size_x", 20)
        self.grid_size_y = kwargs.get("grid_size_y", 20)
        self.grid_size_z = kwargs.get("grid_size_z", 5)

        # ── UAV Start Position (x, y, z) ──
        self.start_pos = kwargs.get("start_pos", (0, 0, 0))

        # ── Waypoints (x, y, z) ──
        default_wp = self._default_waypoints()
        self.waypoints = kwargs.get("waypoints", default_wp)

        # ── Obstacles (건물형: 지상(z=0)에서 랜덤 높이) ──
        self.obstacle_mode = kwargs.get("obstacle_mode", "fixed")
        self.fixed_obstacles = kwargs.get("fixed_obstacles", None)
        self.num_buildings = kwargs.get("num_buildings", 25)
        self.min_building_height = kwargs.get("min_building_height", 1)
        self.max_building_height = kwargs.get("max_building_height", None)  # None → grid_size_z - 1
        self.random_seed = kwargs.get("random_seed", 42)

        # buildings: {(x, y): height} — 시각화용
        self._buildings = {}
        if self.fixed_obstacles is None:
            self.fixed_obstacles = self._default_obstacles()

        # ── Energy ──
        self.energy_budget_multiplier = kwargs.get("energy_budget_multiplier", 2.0)
        self.move_cost = kwargs.get("move_cost", 1.0)
        self.move_cost_z = kwargs.get("move_cost_z", 1.5)  # 수직 이동은 더 비쌈

        # ── Rewards ──
        self.reward_waypoint = kwargs.get("reward_waypoint", 100.0)
        self.reward_all_done = kwargs.get("reward_all_done", 200.0)
        self.penalty_wall = kwargs.get("penalty_wall", -5.0)
        self.penalty_obstacle = kwargs.get("penalty_obstacle", -10.0)
        self.penalty_step = kwargs.get("penalty_step", -1.0)
        self.penalty_energy_out = kwargs.get("penalty_energy_out", -50.0)

        # ── Reward Shaping ──
        self.gamma_shaping = kwargs.get("gamma_shaping", 0.99)

        # ── Max Steps ──
        vol = self.grid_size_x * self.grid_size_y * self.grid_size_z
        self.max_steps = kwargs.get("max_steps", vol)

    # ────────────────────────────────────────────────────────────────────
    def _default_waypoints(self):
        mx = self.grid_size_x // 2
        my = self.grid_size_y // 2
        mz = self.grid_size_z // 2
        return [
            (mx, my, mz),
            (self.grid_size_x - 1, self.grid_size_y - 1, self.grid_size_z - 1),
        ]

    def _default_obstacles(self):
        """건물형 장애물 생성: (x,y) 위치에 z=0부터 랜덤 높이까지 채움"""
        rng = np.random.RandomState(self.random_seed)
        max_h = self.max_building_height or (self.grid_size_z - 1)
        min_h = self.min_building_height

        protected_xy = set()
        for pos3d in [self.start_pos] + list(self._default_waypoints()):
            protected_xy.add((pos3d[0], pos3d[1]))
        protected_3d = {tuple(self.start_pos)}
        for wp in self._default_waypoints():
            protected_3d.add(tuple(wp))

        buildings = {}
        while len(buildings) < self.num_buildings:
            bx = rng.randint(0, self.grid_size_x)
            by = rng.randint(0, self.grid_size_y)
            if (bx, by) in protected_xy or (bx, by) in buildings:
                continue
            height = rng.randint(min_h, max_h + 1)
            buildings[(bx, by)] = height

        # 건물 셀 → 장애물 set 변환 (보호된 3D 좌표 제외)
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
        # random 모드도 건물형으로 생성
        rng = np.random.RandomState(self.random_seed)
        max_h = self.max_building_height or (self.grid_size_z - 1)
        min_h = self.min_building_height

        protected_xy = set()
        for pos3d in [self.start_pos] + list(self.waypoints):
            protected_xy.add((pos3d[0], pos3d[1]))
        protected_3d = {tuple(self.start_pos)} | {tuple(wp) for wp in self.waypoints}

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
        """건물 정보 반환: {(x, y): height}"""
        if not self._buildings:
            # get_obstacles가 아직 호출 안 된 경우 트리거
            self.get_obstacles()
        return dict(self._buildings)

    def compute_energy_budget(self):
        points = [self.start_pos] + list(self.waypoints)
        total_dist = 0
        for i in range(len(points) - 1):
            dx = abs(points[i + 1][0] - points[i][0])
            dy = abs(points[i + 1][1] - points[i][1])
            dz = abs(points[i + 1][2] - points[i][2])
            total_dist += (dx + dy) * self.move_cost + dz * self.move_cost_z
        return total_dist * self.energy_budget_multiplier
