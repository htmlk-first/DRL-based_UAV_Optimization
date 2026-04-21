import numpy as np


class EnvConfig:
    def __init__(self, **kwargs):
        self.grid_size = kwargs.get("grid_size", 100)
        self.start_pos = kwargs.get("start_pos", (0.5, 0.5))

        default_wp = self._default_waypoints(self.grid_size)
        self.waypoints = kwargs.get("waypoints", default_wp)

        self.max_step_size = kwargs.get("max_step_size", 2.2)
        self.wp_reach_radius = kwargs.get("wp_reach_radius", 1.8)

        self.obstacle_mode = kwargs.get("obstacle_mode", "fixed")
        self.fixed_obstacles = kwargs.get("fixed_obstacles", None)

        self.obstacle_footprint_x = int(kwargs.get(
            "obstacle_footprint_x",
            kwargs.get("obstacle_footprint_size", 3),
        ))
        self.obstacle_footprint_y = int(kwargs.get(
            "obstacle_footprint_y",
            kwargs.get("obstacle_footprint_size", 3),
        ))
        self.obstacle_footprint_x = max(1, self.obstacle_footprint_x)
        self.obstacle_footprint_y = max(1, self.obstacle_footprint_y)
        self.obstacle_ground_coverage = kwargs.get("obstacle_ground_coverage", 0.10)

        footprint_area = self.obstacle_footprint_x * self.obstacle_footprint_y
        default_num_obstacles = max(
            1,
            int(round(
                self.grid_size * self.grid_size
                * self.obstacle_ground_coverage / footprint_area
            )),
        )
        self.num_random_obstacles = kwargs.get(
            "num_random_obstacles",
            default_num_obstacles,
        )
        self.random_seed = kwargs.get("random_seed", 42)
        self._obstacle_blocks = {}

        if self.fixed_obstacles is None:
            self.fixed_obstacles = self._default_obstacles(self.grid_size)

        self.energy_budget_multiplier = kwargs.get("energy_budget_multiplier", 3.4)

        self.reward_waypoint = kwargs.get("reward_waypoint", 100.0)
        self.reward_all_done = kwargs.get("reward_all_done", 300.0)
        self.penalty_wall = kwargs.get("penalty_wall", -5.0)
        self.penalty_obstacle = kwargs.get("penalty_obstacle", -10.0)
        self.penalty_step = kwargs.get("penalty_step", -0.5)
        self.penalty_energy_out = kwargs.get("penalty_energy_out", -50.0)

        self.gamma_shaping = kwargs.get("gamma_shaping", 0.99)
        self.max_steps = kwargs.get("max_steps", 1200)

    def _default_waypoints(self, size):
        third = size // 3
        return [
            (third + 0.5, third + 0.5),
            (third * 2 + 0.5, third * 2 + 0.5),
            (size - 0.5, size - 0.5),
        ]

    def _protected_cells(self):
        protected = {(int(self.start_pos[0]), int(self.start_pos[1]))}
        for wp in self.waypoints:
            protected.add((int(wp[0]), int(wp[1])))
        return protected

    def _footprint_cells(self, x0, y0):
        for x in range(x0, x0 + self.obstacle_footprint_x):
            for y in range(y0, y0 + self.obstacle_footprint_y):
                yield x, y

    def _generate_obstacles(self):
        rng = np.random.RandomState(self.random_seed)
        protected = self._protected_cells()
        occupied = set()
        blocks = {}

        max_x0 = self.grid_size - self.obstacle_footprint_x
        max_y0 = self.grid_size - self.obstacle_footprint_y
        if max_x0 < 0 or max_y0 < 0:
            raise ValueError("obstacle footprint is larger than the grid")

        max_attempts = max(1000, self.num_random_obstacles * 300)
        attempts = 0
        while len(blocks) < self.num_random_obstacles and attempts < max_attempts:
            attempts += 1
            x0 = rng.randint(0, max_x0 + 1)
            y0 = rng.randint(0, max_y0 + 1)
            footprint = set(self._footprint_cells(x0, y0))
            if footprint & protected or footprint & occupied:
                continue

            cx = x0 + (self.obstacle_footprint_x - 1) / 2.0
            cy = y0 + (self.obstacle_footprint_y - 1) / 2.0
            blocks[(cx, cy)] = {
                "x0": int(x0),
                "y0": int(y0),
                "size_x": int(self.obstacle_footprint_x),
                "size_y": int(self.obstacle_footprint_y),
            }
            occupied.update(footprint)

        if len(blocks) < self.num_random_obstacles:
            raise RuntimeError(
                "could not place all obstacles; reduce num_random_obstacles "
                "or obstacle footprint"
            )

        self._obstacle_blocks = blocks
        return list(occupied)

    def _default_obstacles(self, size):
        return self._generate_obstacles()

    def get_obstacles(self):
        if self.obstacle_mode == "fixed":
            return list(self.fixed_obstacles)
        return self._generate_obstacles()

    def get_obstacle_blocks(self):
        if not self._obstacle_blocks:
            self.get_obstacles()
        return dict(self._obstacle_blocks)

    def compute_energy_budget(self):
        points = [self.start_pos] + list(self.waypoints)
        total_dist = 0
        for i in range(len(points) - 1):
            total_dist += np.sqrt(
                (points[i + 1][0] - points[i][0]) ** 2
                + (points[i + 1][1] - points[i][1]) ** 2
            )
        return total_dist * self.energy_budget_multiplier
