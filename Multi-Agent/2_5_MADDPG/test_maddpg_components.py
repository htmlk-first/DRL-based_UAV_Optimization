"""Focused regression tests for MADDPG stabilization."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from env.config import MultiAgentEnvConfig
from env.multi_uav_env import MultiUAVEnv
from maddpg_agent import MADDPGAgent
from train import team_rewards


class MADDPGComponentTests(unittest.TestCase):
    def test_actions_are_projected_to_unit_disk(self):
        actions = np.array([[1.0, 1.0], [0.2, -0.3]], dtype=np.float32)
        projected = MADDPGAgent.project_actions(actions)

        self.assertLessEqual(float(np.linalg.norm(projected[0])), 1.0 + 1e-6)
        np.testing.assert_allclose(projected[1], actions[1])

    def test_environment_records_failed_action(self):
        config = MultiAgentEnvConfig(
            grid_size=5,
            n_agents=1,
            start_positions=[(0.0, 0.0)],
            waypoints=[(4.0, 4.0)],
            fixed_obstacles=[],
            max_steps=5,
        )
        env = MultiUAVEnv(config)
        initial_obs, _ = env.reset()
        next_obs, _, _, _, _ = env.step([[-1.0, 0.0]])

        self.assertEqual(float(initial_obs[0, -1]), 0.0)
        self.assertEqual(float(next_obs[0, -1]), 1.0)
        np.testing.assert_allclose(next_obs[0, -3:-1], [-1.0, 0.0])

    def test_movement_segment_cannot_cut_through_obstacle_corner(self):
        config = MultiAgentEnvConfig(
            grid_size=5,
            n_agents=1,
            start_positions=[(0.4, 0.8)],
            waypoints=[(4.0, 4.0)],
            fixed_obstacles=[(1, 1)],
            max_steps=5,
        )
        env = MultiUAVEnv(config)
        env.reset()
        _, rewards, _, _, info = env.step([[0.4, -0.4]])

        np.testing.assert_allclose(env.positions[0], [0.4, 0.8])
        self.assertEqual(info["events_by_agent"][0], "obstacle_collision")
        self.assertLess(float(rewards[0]), 0.0)

    def test_guided_path_does_not_intersect_rendered_obstacles(self):
        config = MultiAgentEnvConfig(
            grid_size=5,
            n_agents=1,
            start_positions=[(0.0, 0.0)],
            waypoints=[(2.0, 2.0)],
            fixed_obstacles=[(1, 1)],
            max_steps=20,
            wp_reach_radius=0.9,
        )
        env = MultiUAVEnv(config)
        env.reset()
        done = False

        while not done:
            _, _, terminated, truncated, _ = env.step(
                env.get_guided_actions()
            )
            done = terminated or truncated

        for start, end in zip(env.paths[0], env.paths[0][1:]):
            self.assertFalse(env._segment_hits_obstacle(start, end))

    def test_team_reward_is_shared(self):
        rewards = team_rewards(np.array([4.0, -2.0, 2.0, 0.0]))
        np.testing.assert_allclose(rewards, np.ones(4))

    def test_guidance_assigns_distinct_targets_and_completes(self):
        config = MultiAgentEnvConfig(
            grid_size=5,
            n_agents=2,
            start_positions=[(0.0, 0.0), (4.0, 4.0)],
            waypoints=[(1.0, 1.0), (3.0, 3.0)],
            fixed_obstacles=[],
            max_steps=20,
            wp_reach_radius=0.9,
        )
        env = MultiUAVEnv(config)
        env.reset()

        assignments = env._assigned_targets()
        self.assertEqual(len(set(assignments)), 2)

        done = False
        info = {}
        while not done:
            _, _, terminated, truncated, info = env.step(
                env.get_guided_actions()
            )
            done = terminated or truncated

        self.assertEqual(info["event"], "mission_complete")

    def test_warmup_interval_and_delayed_actor_update(self):
        agent = MADDPGAgent(
            obs_dim=4,
            action_dim=2,
            n_agents=2,
            batch_size=2,
            warmup_steps=4,
            learn_every=2,
            policy_delay=2,
            hidden_dim=16,
            device=torch.device("cpu"),
        )
        obs = np.zeros((2, 4), dtype=np.float32)
        actions = np.zeros((2, 2), dtype=np.float32)
        rewards = np.zeros(2, dtype=np.float32)

        for _ in range(3):
            agent.store_joint(obs, actions, rewards, obs, False)
            self.assertEqual(agent.learn(), (None, None))

        agent.store_joint(obs, actions, rewards, obs, True)
        critic_loss, actor_loss = agent.learn()
        self.assertIsNotNone(critic_loss)
        self.assertIsNone(actor_loss)

        agent.store_joint(obs, actions, rewards, obs, True)
        self.assertEqual(agent.learn(), (None, None))
        agent.store_joint(obs, actions, rewards, obs, True)
        critic_loss, actor_loss = agent.learn()
        self.assertIsNotNone(critic_loss)
        self.assertIsNotNone(actor_loss)


if __name__ == "__main__":
    unittest.main()
