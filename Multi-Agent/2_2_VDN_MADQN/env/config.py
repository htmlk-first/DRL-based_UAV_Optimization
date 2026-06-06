"""Configuration for the cooperative multi-agent DQN UAV environment."""

from __future__ import annotations

import numpy as np


class MultiAgentEnvConfig:
    def __init__(self, **kwargs):
        self.grid_size = kwargs.get("grid_size", 20)
        self.n_agents = kwargs.get("n_agents", 4)

        default_starts = self._default_start_positions(self.n_agents)
        self.start_positions = [tuple(p) for p in kwargs.get("start_positions", default_starts)]
        if len(self.start_positions) != self.n_agents:
            raise ValueError("start_positions length must match n_agents")

        default_wp = self._default_waypoints(self.grid_size)
        self.waypoints = [tuple(wp) for wp in kwargs.get("waypoints", default_wp)]
        self.ordered_waypoints = kwargs.get("ordered_waypoints", False)

        self.obstacle_mode = kwargs.get("obstacle_mode", "fixed")
        self.fixed_obstacles = kwargs.get("fixed_obstacles", None)
        self.num_random_obstacles = kwargs.get("num_random_obstacles", 15)
        self.random_seed = kwargs.get("random_seed", 42)

        if self.fixed_obstacles is None:
            self.fixed_obstacles = self._default_obstacles(self.grid_size)

        self.energy_budget_multiplier = kwargs.get("energy_budget_multiplier", 2.2)
        self.move_cost = kwargs.get("move_cost", 1.0)

        self.reward_waypoint = kwargs.get("reward_waypoint", 100.0)
        self.reward_all_done = kwargs.get("reward_all_done", 200.0)
        self.penalty_wall = kwargs.get("penalty_wall", -5.0)
        self.penalty_obstacle = kwargs.get("penalty_obstacle", -10.0)
        self.penalty_step = kwargs.get("penalty_step", -1.0)
        self.penalty_energy_out = kwargs.get("penalty_energy_out", -50.0)
        self.collision_penalty = kwargs.get("collision_penalty", -20.0)

        self.gamma_shaping = kwargs.get("gamma_shaping", 0.99)
        self.max_steps = kwargs.get("max_steps", int(self.grid_size * self.grid_size * 2.5))

    def _default_start_positions(self, n_agents):
        corners = [
            (0, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, 0),
            (self.grid_size - 1, self.grid_size - 1),
        ]
        if n_agents <= len(corners):
            return corners[:n_agents]

        starts = list(corners)
        for i in range(n_agents - len(corners)):
            starts.append((0, min(i + 1, self.grid_size - 2)))
        return starts

    def _default_waypoints(self, size):
        return [
            (4, 4),
            (4, 10),
            (4, 15),
            (8, 6),
            (8, 13),
            (12, 4),
            (12, 10),
            (12, 16),
            (16, 7),
            (16, 14),
        ]

    def _protected_cells(self):
        return set(self.start_positions) | set(self.waypoints)

    def _default_obstacles(self, size):
        rng = np.random.RandomState(42)
        n = max(10, size * size // 10)
        obstacles = set()
        protected = self._protected_cells()

        while len(obstacles) < n:
            pos = (rng.randint(0, size), rng.randint(0, size))
            if pos not in protected:
                obstacles.add(pos)
        return list(obstacles)

    def get_obstacles(self):
        if self.obstacle_mode == "fixed":
            return list(self.fixed_obstacles)

        rng = np.random.RandomState(self.random_seed)
        obstacles = set()
        protected = self._protected_cells()
        while len(obstacles) < self.num_random_obstacles:
            pos = (rng.randint(0, self.grid_size), rng.randint(0, self.grid_size))
            if pos not in protected:
                obstacles.add(pos)
        return list(obstacles)

    def compute_energy_budget(self):
        longest_route = 0
        for start in self.start_positions:
            points = [start] + list(self.waypoints)
            total_dist = 0
            for i in range(len(points) - 1):
                total_dist += abs(points[i + 1][0] - points[i][0])
                total_dist += abs(points[i + 1][1] - points[i][1])
            longest_route = max(longest_route, total_dist)
        return longest_route * self.energy_budget_multiplier
