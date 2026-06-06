"""Cooperative multi-UAV 2D environment for MADDPG continuous control."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import MultiAgentEnvConfig


class MultiUAVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ADJ_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else MultiAgentEnvConfig()
        self.grid_size = self.config.grid_size
        self.n_agents = self.config.n_agents
        self.waypoints = [tuple(wp) for wp in self.config.waypoints]
        self.n_waypoints = len(self.waypoints)
        self.obstacles = set(map(tuple, self.config.get_obstacles()))
        self.energy_budget = self.config.compute_energy_budget()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, 2),
            dtype=np.float32,
        )

        obs_dim = (
            2
            + 1
            + self.n_waypoints
            + 4
            + 2
            + self.n_agents
            + 2 * (self.n_agents - 1)
            + 2
            + 1
        )
        self.single_observation_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.ones((self.n_agents, obs_dim), dtype=np.float32),
            high=np.ones((self.n_agents, obs_dim), dtype=np.float32),
            dtype=np.float32,
        )

        self.positions = None
        self.energies = None
        self.visited = None
        self.steps = None
        self.paths = None
        self.collisions = None
        self._prev_potential = None
        self.last_actions = None
        self.last_action_failed = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = [list(map(float, pos)) for pos in self.config.start_positions]
        self.energies = [float(self.energy_budget) for _ in range(self.n_agents)]
        self.visited = [False] * self.n_waypoints
        self.steps = 0
        self.paths = [[tuple(pos)] for pos in self.positions]
        self.collisions = 0
        self._prev_potential = self._potential()
        self.last_actions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.last_action_failed = np.zeros(self.n_agents, dtype=np.float32)

        return self._get_obs(), self._build_info("reset", ["reset"] * self.n_agents)

    def step(self, actions):
        actions = np.clip(
            np.asarray(actions, dtype=np.float32).reshape(self.n_agents, 2),
            -1.0,
            1.0,
        )
        action_norms = np.linalg.norm(actions, axis=1, keepdims=True)
        actions = actions / np.maximum(action_norms, 1.0)

        self.steps += 1
        old_positions = [tuple(pos) for pos in self.positions]
        proposed = []
        move_dists = []
        base_valid = [True] * self.n_agents
        events_by_agent = ["move"] * self.n_agents
        rewards = np.full(self.n_agents, self.config.penalty_step, dtype=np.float32)

        for i, action in enumerate(actions):
            dx = float(action[0]) * self.config.max_step_size
            dy = float(action[1]) * self.config.max_step_size
            move_dist = float(np.sqrt(dx * dx + dy * dy))
            nx = old_positions[i][0] + dx
            ny = old_positions[i][1] + dy
            proposed_pos = (nx, ny)
            proposed.append(proposed_pos)
            move_dists.append(move_dist)

            if not self._position_is_in_bounds(proposed_pos):
                base_valid[i] = False
                events_by_agent[i] = "wall_collision"
                rewards[i] += self.config.penalty_wall
            elif self._segment_hits_obstacle(
                old_positions[i], proposed_pos
            ):
                base_valid[i] = False
                events_by_agent[i] = "obstacle_collision"
                rewards[i] += self.config.penalty_obstacle

        collision_agents = self._collision_agents(old_positions, proposed, base_valid)
        for i in collision_agents:
            rewards[i] += self.config.collision_penalty
            events_by_agent[i] = "uav_collision"

        for i in range(self.n_agents):
            if base_valid[i] and i not in collision_agents:
                self.positions[i] = [proposed[i][0], proposed[i][1]]
                self.energies[i] -= move_dists[i] * self.config.move_cost

        for i in range(self.n_agents):
            self.paths[i].append(tuple(self.positions[i]))

        step_collisions = len(collision_agents)
        self.collisions += step_collisions

        waypoint_reward = self._apply_waypoint_rewards(events_by_agent)
        team_reward = waypoint_reward
        completion_reward = 0.0
        terminated = False
        truncated = False
        event = self._top_event(events_by_agent)

        if all(self.visited):
            completion_reward = float(self.config.reward_all_done)
            team_reward += completion_reward
            terminated = True
            event = "mission_complete"

        new_potential = self._potential()
        shaping = self.config.gamma_shaping * new_potential - self._prev_potential
        team_reward += shaping
        self._prev_potential = new_potential
        rewards += team_reward

        if not terminated:
            depleted = [i for i, e in enumerate(self.energies) if e <= 0]
            if depleted:
                for i in depleted:
                    rewards[i] += self.config.penalty_energy_out
                    events_by_agent[i] = "energy_depleted"
                terminated = True
                event = "energy_depleted"

        if not terminated and self.steps >= self.config.max_steps:
            truncated = True
            event = "max_steps_reached"

        self.last_actions = actions.copy()
        self.last_action_failed = np.array([
            float((not base_valid[i]) or (i in collision_agents))
            for i in range(self.n_agents)
        ], dtype=np.float32)

        info = self._build_info(event, events_by_agent)
        info["step_collisions"] = step_collisions
        info["team_reward"] = float(team_reward)
        info["waypoint_reward"] = float(waypoint_reward)
        info["completion_reward"] = float(completion_reward)
        info["shaping_reward"] = float(shaping)

        return self._get_obs(), rewards.astype(np.float32), terminated, truncated, info

    def _collision_agents(self, old_positions, proposed, base_valid):
        collision_agents = set()
        radius = float(self.config.collision_radius)

        for i in range(self.n_agents):
            if not base_valid[i]:
                continue
            for j in range(i + 1, self.n_agents):
                if not base_valid[j]:
                    continue
                if self._distance(proposed[i], proposed[j]) < radius:
                    collision_agents.add(i)
                    collision_agents.add(j)
                elif (
                    self._distance(proposed[i], old_positions[j]) < radius
                    and self._distance(proposed[j], old_positions[i]) < radius
                ):
                    collision_agents.add(i)
                    collision_agents.add(j)

        changed = True
        while changed:
            changed = False
            stationary = {
                j for j in range(self.n_agents)
                if (not base_valid[j]) or (j in collision_agents)
            }
            for i in range(self.n_agents):
                if (not base_valid[i]) or (i in collision_agents):
                    continue
                for j in stationary:
                    if i != j and self._distance(proposed[i], old_positions[j]) < radius:
                        collision_agents.add(i)
                        changed = True
                        break

        return collision_agents

    def _apply_waypoint_rewards(self, events_by_agent):
        if not self.config.ordered_waypoints:
            team_reward = 0.0
            for target_idx, target in enumerate(self.waypoints):
                if self.visited[target_idx]:
                    continue
                for i, pos in enumerate(self.positions):
                    if self._distance(pos, target) <= self.config.wp_reach_radius:
                        self.visited[target_idx] = True
                        events_by_agent[i] = f"waypoint_{target_idx}_reached"
                        team_reward += float(self.config.reward_waypoint)
                        break
            return team_reward

        target_idx = self._next_target_index()
        if target_idx is None:
            return 0.0

        target = self.waypoints[target_idx]
        for i, pos in enumerate(self.positions):
            if self._distance(pos, target) <= self.config.wp_reach_radius:
                self.visited[target_idx] = True
                events_by_agent[i] = f"waypoint_{target_idx}_reached"
                return float(self.config.reward_waypoint)
        return 0.0

    def _top_event(self, events_by_agent):
        for event in events_by_agent:
            if event.startswith("waypoint_"):
                return event
        for event in ("uav_collision", "obstacle_collision", "wall_collision"):
            if event in events_by_agent:
                return event
        return "move"

    def _build_info(self, event, events_by_agent):
        return {
            "event": event,
            "events_by_agent": list(events_by_agent),
            "positions": [tuple(pos) for pos in self.positions],
            "energies": [float(e) for e in self.energies],
            "visited": list(self.visited),
            "steps": self.steps,
            "collisions": int(self.collisions),
            "waypoints_visited": int(sum(self.visited)),
        }

    def _get_obs(self):
        size = max(self.grid_size - 1, 1)
        occupied = {self._cell(pos) for pos in self.positions}
        obs_batch = []
        assigned_targets = self._assigned_targets()

        for agent_idx, pos in enumerate(self.positions):
            x, y = pos
            cell_x, cell_y = self._cell(pos)
            adj = []
            for dx, dy in self.ADJ_OFFSETS:
                neighbor = (cell_x + dx, cell_y + dy)
                nx, ny = neighbor
                if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                    adj.append(1.0)
                elif neighbor in self.obstacles or neighbor in (occupied - {self._cell(pos)}):
                    adj.append(1.0)
                else:
                    adj.append(0.0)

            target = assigned_targets[agent_idx]
            dx_wp = (target[0] - x) / size
            dy_wp = (target[1] - y) / size
            agent_id = [0.0] * self.n_agents
            agent_id[agent_idx] = 1.0

            rel_positions = []
            for other_idx, other_pos in enumerate(self.positions):
                if other_idx == agent_idx:
                    continue
                rel_positions.extend([
                    (other_pos[0] - x) / size,
                    (other_pos[1] - y) / size,
                ])

            obs = (
                [x / size, y / size, self.energies[agent_idx] / max(self.energy_budget, 1e-8)]
                + [float(v) for v in self.visited]
                + adj
                + [dx_wp, dy_wp]
                + agent_id
                + rel_positions
                + self.last_actions[agent_idx].tolist()
                + [float(self.last_action_failed[agent_idx])]
            )
            obs_batch.append(obs)

        return np.array(obs_batch, dtype=np.float32)

    def _assigned_targets(self):
        """Assign distinct nearby waypoints to agents whenever possible."""
        unvisited = [
            wp for i, wp in enumerate(self.waypoints)
            if not self.visited[i]
        ]
        if not unvisited:
            return [self.waypoints[-1]] * self.n_agents

        assignments = [None] * self.n_agents
        remaining_agents = set(range(self.n_agents))
        remaining_targets = list(unvisited)

        while remaining_agents and remaining_targets:
            agent_idx, target = min(
                (
                    (i, wp)
                    for i in remaining_agents
                    for wp in remaining_targets
                ),
                key=lambda pair: self._distance(
                    self.positions[pair[0]], pair[1]
                ),
            )
            assignments[agent_idx] = target
            remaining_agents.remove(agent_idx)
            remaining_targets.remove(target)

        for agent_idx in remaining_agents:
            assignments[agent_idx] = min(
                unvisited,
                key=lambda wp: self._distance(
                    self.positions[agent_idx], wp
                ),
            )

        return assignments

    def get_guided_actions(self):
        """Return collision-aware waypoint actions for guided exploration."""
        targets = self._assigned_targets()
        actions = []
        reserved_proposals = []
        angles = (
            0.0,
            math.pi / 4,
            -math.pi / 4,
            math.pi / 2,
            -math.pi / 2,
            3 * math.pi / 4,
            -3 * math.pi / 4,
            math.pi,
        )

        for agent_idx, (pos, target) in enumerate(
            zip(self.positions, targets)
        ):
            direction = np.asarray(target, dtype=np.float32) - np.asarray(
                pos, dtype=np.float32
            )
            norm = float(np.linalg.norm(direction))
            base = direction / max(norm, 1e-8)
            selected = np.zeros(2, dtype=np.float32)

            for angle in angles:
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                candidate = np.array([
                    base[0] * cos_a - base[1] * sin_a,
                    base[0] * sin_a + base[1] * cos_a,
                ], dtype=np.float32)
                proposal = (
                    np.asarray(pos, dtype=np.float32)
                    + candidate * self.config.max_step_size
                )
                if self._guided_proposal_is_valid(
                    agent_idx, proposal, reserved_proposals
                ):
                    selected = candidate
                    reserved_proposals.append(proposal)
                    break
            else:
                reserved_proposals.append(
                    np.asarray(pos, dtype=np.float32)
                )

            actions.append(selected)

        return np.asarray(actions, dtype=np.float32)

    def _guided_proposal_is_valid(
        self,
        agent_idx,
        proposal,
        reserved_proposals,
    ):
        if not self._position_is_in_bounds(proposal):
            return False
        if self._segment_hits_obstacle(
            self.positions[agent_idx], proposal
        ):
            return False

        radius = float(self.config.collision_radius)
        for other_idx, other_pos in enumerate(self.positions):
            if (
                other_idx != agent_idx
                and self._distance(proposal, other_pos) < radius
            ):
                return False
        return not any(
            self._distance(proposal, other) < radius
            for other in reserved_proposals
        )

    def _next_target_index(self):
        for i, visited in enumerate(self.visited):
            if not visited:
                return i
        return None

    def _next_target(self):
        idx = self._next_target_index()
        if idx is None:
            return self.waypoints[-1]
        return self.waypoints[idx]

    def _target_for_agent(self, agent_idx):
        if self.config.ordered_waypoints:
            return self._next_target()
        return self._assigned_targets()[agent_idx]

    def _potential(self):
        if self.config.ordered_waypoints:
            idx = self._next_target_index()
            if idx is None:
                return 0.0
            target = self.waypoints[idx]
            return -float(min(self._distance(pos, target) for pos in self.positions))

        unvisited = [
            wp for i, wp in enumerate(self.waypoints)
            if not self.visited[i]
        ]
        if not unvisited:
            return 0.0

        total = 0.0
        for wp in unvisited:
            total += min(self._distance(pos, wp) for pos in self.positions)
        return -float(total)

    def _cell(self, pos):
        # Integer map coordinates denote cell centers in rendering and config.
        x = int(np.floor(pos[0] + 0.5))
        y = int(np.floor(pos[1] + 0.5))
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))
        return x, y

    def _position_is_in_bounds(self, pos):
        limit = self.grid_size - 1
        return 0.0 <= pos[0] <= limit and 0.0 <= pos[1] <= limit

    def _segment_hits_obstacle(self, start, end):
        return any(
            self._segment_intersects_cell(start, end, obstacle)
            for obstacle in self.obstacles
        )

    @staticmethod
    def _segment_intersects_cell(start, end, cell):
        """Test a movement segment against a rendered unit obstacle square."""
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        lower = np.asarray(cell, dtype=np.float64) - 0.5
        upper = np.asarray(cell, dtype=np.float64) + 0.5
        direction = end - start
        t_min = 0.0
        t_max = 1.0

        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < lower[axis] or start[axis] > upper[axis]:
                    return False
                continue

            enter = (lower[axis] - start[axis]) / direction[axis]
            exit_ = (upper[axis] - start[axis]) / direction[axis]
            if enter > exit_:
                enter, exit_ = exit_, enter
            t_min = max(t_min, enter)
            t_max = min(t_max, exit_)
            if t_min > t_max:
                return False

        return True

    @staticmethod
    def _distance(a, b):
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    def get_grid_map(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 1
        for i, wp in enumerate(self.waypoints):
            cell = self._cell(wp)
            grid[cell[0], cell[1]] = 5 if self.visited[i] else 2
        for i, pos in enumerate(self.positions):
            cell = self._cell(pos)
            grid[cell[0], cell[1]] = 10 + i
        return grid

    def render(self, mode="human"):
        grid = self.get_grid_map()
        symbols = {
            0: "..", 1: "##", 2: "WP", 5: "OK",
            10: "A0", 11: "A1", 12: "A2", 13: "A3",
        }
        print(f"\nStep: {self.steps} | Visited: {self.visited} | Collisions: {self.collisions}")
        print(f"Positions: {[tuple(round(v, 2) for v in p) for p in self.positions]}")
        print("+" + "--" * self.grid_size + "+")
        for row in grid:
            line = "|" + "".join(symbols.get(int(c), "AA") for c in row) + "|"
            print(line)
        print("+" + "--" * self.grid_size + "+")
