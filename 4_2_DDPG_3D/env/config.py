import numpy as np


class EnvConfig3D:
    def __init__(self, **kwargs):
        self.grid_size_x = kwargs.get("grid_size_x", 100)
        self.grid_size_y = kwargs.get("grid_size_y", 100)
        self.grid_size_z = kwargs.get("grid_size_z", 10)

        self.start_pos = kwargs.get("start_pos", (0.5, 0.5, 0.5))

        default_wp = self._default_waypoints()
        self.waypoints = kwargs.get("waypoints", default_wp)

        self.max_step_size = kwargs.get("max_step_size", 2.2)
        self.max_step_size_z = kwargs.get("max_step_size_z", 1.0)
        self.wp_reach_radius = kwargs.get("wp_reach_radius", 2.0)

        self.obstacle_mode = kwargs.get("obstacle_mode", "fixed")
        self.fixed_obstacles = kwargs.get("fixed_obstacles", None)

        self.building_footprint_x = int(kwargs.get(
            "building_footprint_x",
            kwargs.get("building_footprint_size", 3),
        ))
        self.building_footprint_y = int(kwargs.get(
            "building_footprint_y",
            kwargs.get("building_footprint_size", 3),
        ))
        self.building_footprint_x = max(1, self.building_footprint_x)
        self.building_footprint_y = max(1, self.building_footprint_y)
        self.building_ground_coverage = kwargs.get("building_ground_coverage", 0.044)

        footprint_area = self.building_footprint_x * self.building_footprint_y
        default_num_buildings = max(
            10,
            int(round(
                self.grid_size_x * self.grid_size_y
                * self.building_ground_coverage / footprint_area
            )),
        )
        self.num_buildings = kwargs.get("num_buildings", default_num_buildings)
        self.min_building_height = kwargs.get("min_building_height", 1)
        self.max_building_height = kwargs.get("max_building_height", None)
        self.random_seed = kwargs.get("random_seed", 42)

        self._buildings = {}
        if self.fixed_obstacles is None:
            self.fixed_obstacles = self._default_obstacles()

        self.energy_budget_multiplier = kwargs.get("energy_budget_multiplier", 3.6)
        self.z_cost_multiplier = kwargs.get("z_cost_multiplier", 1.7)

        self.reward_waypoint = kwargs.get("reward_waypoint", 100.0)
        self.reward_all_done = kwargs.get("reward_all_done", 300.0)
        self.penalty_wall = kwargs.get("penalty_wall", -5.0)
        self.penalty_obstacle = kwargs.get("penalty_obstacle", -10.0)
        self.penalty_step = kwargs.get("penalty_step", -0.5)
        self.penalty_energy_out = kwargs.get("penalty_energy_out", -50.0)

        self.gamma_shaping = kwargs.get("gamma_shaping", 0.99)
        self.max_steps = kwargs.get("max_steps", 1500)

    def _default_waypoints(self):
        third_x = self.grid_size_x // 3
        third_y = self.grid_size_y // 3
        mz = self.grid_size_z // 2
        return [
            (third_x + 0.5, third_y + 0.5, mz + 0.5),
            (third_x * 2 + 0.5, third_y * 2 + 0.5, mz + 0.5),
            (self.grid_size_x - 0.5, self.grid_size_y - 0.5,
             self.grid_size_z - 0.5),
        ]

    def _protected_cells(self, waypoints):
        protected_xy = set()
        protected_3d = {
            (int(self.start_pos[0]), int(self.start_pos[1]),
             int(self.start_pos[2]))
        }

        for pos3d in [self.start_pos] + list(waypoints):
            protected_xy.add((int(pos3d[0]), int(pos3d[1])))

        for wp in waypoints:
            protected_3d.add((int(wp[0]), int(wp[1]), int(wp[2])))

        return protected_xy, protected_3d

    def _footprint_cells(self, x0, y0):
        for bx in range(x0, x0 + self.building_footprint_x):
            for by in range(y0, y0 + self.building_footprint_y):
                yield bx, by

    def _generate_building_obstacles(self, waypoints):
        rng = np.random.RandomState(self.random_seed)
        max_h = self.max_building_height or (self.grid_size_z - 1)
        min_h = self.min_building_height

        protected_xy, protected_3d = self._protected_cells(waypoints)
        occupied_xy = set()
        buildings = {}

        max_x0 = self.grid_size_x - self.building_footprint_x
        max_y0 = self.grid_size_y - self.building_footprint_y
        if max_x0 < 0 or max_y0 < 0:
            raise ValueError("building footprint is larger than the grid")

        max_attempts = max(1000, self.num_buildings * 300)
        attempts = 0
        while len(buildings) < self.num_buildings and attempts < max_attempts:
            attempts += 1
            x0 = rng.randint(0, max_x0 + 1)
            y0 = rng.randint(0, max_y0 + 1)
            footprint = set(self._footprint_cells(x0, y0))
            if footprint & protected_xy or footprint & occupied_xy:
                continue

            height = int(rng.randint(min_h, max_h + 1))
            cx = x0 + (self.building_footprint_x - 1) / 2.0
            cy = y0 + (self.building_footprint_y - 1) / 2.0
            buildings[(cx, cy)] = {
                "height": height,
                "x0": int(x0),
                "y0": int(y0),
                "size_x": int(self.building_footprint_x),
                "size_y": int(self.building_footprint_y),
            }
            occupied_xy.update(footprint)

        if len(buildings) < self.num_buildings:
            raise RuntimeError(
                "could not place all buildings; reduce num_buildings or footprint"
            )

        obstacles = set()
        for meta in buildings.values():
            h = meta["height"]
            for bx in range(meta["x0"], meta["x0"] + meta["size_x"]):
                for by in range(meta["y0"], meta["y0"] + meta["size_y"]):
                    for bz in range(h):
                        cell = (bx, by, bz)
                        if cell not in protected_3d:
                            obstacles.add(cell)

        self._buildings = buildings
        return list(obstacles)

    def _default_obstacles(self):
        return self._generate_building_obstacles(self.waypoints)

    def get_obstacles(self):
        if self.obstacle_mode == "fixed":
            return list(self.fixed_obstacles)
        return self._generate_building_obstacles(self.waypoints)

    def get_buildings(self):
        if not self._buildings:
            self.get_obstacles()
        return dict(self._buildings)

    def compute_energy_budget(self):
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
