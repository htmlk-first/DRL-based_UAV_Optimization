"""Cooperative multi-UAV grid environment for QMIX-MADDQN."""

from __future__ import annotations

from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import MultiAgentEnvConfig


class MultiUAVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else MultiAgentEnvConfig()
        self.grid_size = self.config.grid_size
        self.n_agents = self.config.n_agents
        self.waypoints = [tuple(wp) for wp in self.config.waypoints]
        self.n_waypoints = len(self.waypoints)
        self.obstacles = set(map(tuple, self.config.get_obstacles()))
        self.energy_budget = self.config.compute_energy_budget()

        self.action_space = spaces.MultiDiscrete([4] * self.n_agents)

        obs_dim = (
            2
            + 1
            + self.n_waypoints
            + 4
            + 2
            + self.n_agents
            + 2 * (self.n_agents - 1)
            + len(self.MOVES)
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
        self.positions = [list(pos) for pos in self.config.start_positions]
        self.energies = [float(self.energy_budget) for _ in range(self.n_agents)]
        self.visited = [False] * self.n_waypoints
        self.steps = 0
        self.paths = [[tuple(pos)] for pos in self.positions]
        self.collisions = 0
        self._prev_potential = self._potential()
        self.last_actions = np.full(self.n_agents, -1, dtype=np.int64)
        self.last_action_failed = np.zeros(self.n_agents, dtype=np.float32)

        return self._get_obs(), self._build_info("reset", ["reset"] * self.n_agents)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        assert self.action_space.contains(actions), f"Invalid actions: {actions}"

        self.steps += 1
        old_positions = [tuple(pos) for pos in self.positions]
        proposed = []
        base_valid = [True] * self.n_agents
        events_by_agent = ["move"] * self.n_agents
        rewards = np.full(self.n_agents, self.config.penalty_step, dtype=np.float32)

        for i, action in enumerate(actions):
            dx, dy = self.MOVES[int(action)]
            nx = old_positions[i][0] + dx
            ny = old_positions[i][1] + dy
            proposed_pos = (nx, ny)
            proposed.append(proposed_pos)

            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                base_valid[i] = False
                events_by_agent[i] = "wall_collision"
                rewards[i] += self.config.penalty_wall
            elif proposed_pos in self.obstacles:
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
                self.energies[i] -= self.config.move_cost

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

        target_counts = Counter(proposed[i] for i in range(self.n_agents) if base_valid[i])
        for i in range(self.n_agents):
            if base_valid[i] and target_counts[proposed[i]] > 1:
                collision_agents.add(i)

        for i in range(self.n_agents):
            if not base_valid[i]:
                continue
            for j in range(i + 1, self.n_agents):
                if not base_valid[j]:
                    continue
                if proposed[i] == old_positions[j] and proposed[j] == old_positions[i]:
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
                    if i != j and proposed[i] == old_positions[j]:
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
                    if tuple(pos) == target:
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
            if tuple(pos) == target:
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
        occupied = {tuple(pos) for pos in self.positions}
        obs_batch = []

        for agent_idx, pos in enumerate(self.positions):
            x, y = pos
            adj = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                    adj.append(1.0)
                elif neighbor in self.obstacles or neighbor in (occupied - {tuple(pos)}):
                    adj.append(1.0)
                else:
                    adj.append(0.0)

            target = self._target_for_agent(agent_idx)
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

            previous_action = [0.0] * len(self.MOVES)
            if self.last_actions[agent_idx] >= 0:
                previous_action[int(self.last_actions[agent_idx])] = 1.0

            obs = (
                [x / size, y / size, self.energies[agent_idx] / max(self.energy_budget, 1e-8)]
                + [float(v) for v in self.visited]
                + adj
                + [dx_wp, dy_wp]
                + agent_id
                + rel_positions
                + previous_action
                + [float(self.last_action_failed[agent_idx])]
            )
            obs_batch.append(obs)

        return np.array(obs_batch, dtype=np.float32)

    def get_action_masks(self):
        """Return locally safe actions for decentralized action selection."""
        occupied = {tuple(pos) for pos in self.positions}
        masks = np.ones((self.n_agents, self.action_space.nvec[0]), dtype=bool)

        for agent_idx, pos in enumerate(self.positions):
            for action, (dx, dy) in self.MOVES.items():
                neighbor = (pos[0] + dx, pos[1] + dy)
                blocked = (
                    neighbor[0] < 0
                    or neighbor[0] >= self.grid_size
                    or neighbor[1] < 0
                    or neighbor[1] >= self.grid_size
                    or neighbor in self.obstacles
                    or neighbor in (occupied - {tuple(pos)})
                )
                masks[agent_idx, action] = not blocked

            # The environment has no hover action. Keep the policy operable in
            # the rare case that every neighboring cell is blocked.
            if not masks[agent_idx].any():
                masks[agent_idx] = True

        return masks

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

        unvisited = [
            wp for i, wp in enumerate(self.waypoints)
            if not self.visited[i]
        ]
        if not unvisited:
            return self.waypoints[-1]

        x, y = self.positions[agent_idx]
        return min(unvisited, key=lambda wp: abs(x - wp[0]) + abs(y - wp[1]))

    def _potential(self):
        if self.config.ordered_waypoints:
            idx = self._next_target_index()
            if idx is None:
                return 0.0
            target = self.waypoints[idx]
            distances = [
                abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                for pos in self.positions
            ]
            return -float(min(distances))

        unvisited = [
            wp for i, wp in enumerate(self.waypoints)
            if not self.visited[i]
        ]
        if not unvisited:
            return 0.0

        total = 0.0
        for wp in unvisited:
            total += min(
                abs(pos[0] - wp[0]) + abs(pos[1] - wp[1])
                for pos in self.positions
            )
        return -float(total)

    def get_grid_map(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 1
        for i, wp in enumerate(self.waypoints):
            grid[wp[0], wp[1]] = 5 if self.visited[i] else 2
        for i, pos in enumerate(self.positions):
            grid[pos[0], pos[1]] = 10 + i
        return grid

    def render(self, mode="human"):
        grid = self.get_grid_map()
        symbols = {0: "..", 1: "##", 2: "WP", 5: "OK", 10: "A0", 11: "A1", 12: "A2"}
        print(f"\nStep: {self.steps} | Visited: {self.visited} | Collisions: {self.collisions}")
        print(f"Positions: {[tuple(p) for p in self.positions]}")
        print("+" + "--" * self.grid_size + "+")
        for row in grid:
            line = "|" + "".join(symbols.get(int(c), "AA") for c in row) + "|"
            print(line)
        print("+" + "--" * self.grid_size + "+")
