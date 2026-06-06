"""Focused regression tests for QMIX action masking and update scheduling."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from env.config import MultiAgentEnvConfig
from env.multi_uav_env import MultiUAVEnv
from qmix_maddqn_agent import QMixer, QMIXMADDQNAgent


class QMIXComponentTests(unittest.TestCase):
    def test_environment_masks_walls(self):
        config = MultiAgentEnvConfig(
            grid_size=5,
            n_agents=1,
            start_positions=[(0, 0)],
            waypoints=[(4, 4)],
            fixed_obstacles=[],
        )
        env = MultiUAVEnv(config)
        env.reset()

        np.testing.assert_array_equal(
            env.get_action_masks()[0],
            np.array([False, True, False, True]),
        )

    def test_collision_feedback_changes_the_next_observation(self):
        config = MultiAgentEnvConfig(
            grid_size=5,
            n_agents=2,
            start_positions=[(1, 0), (1, 2)],
            waypoints=[(4, 4)],
            fixed_obstacles=[],
        )
        env = MultiUAVEnv(config)
        initial_obs, _ = env.reset()
        next_obs, _, _, _, _ = env.step([
            MultiUAVEnv.ACTION_RIGHT,
            MultiUAVEnv.ACTION_LEFT,
        ])

        np.testing.assert_array_equal(initial_obs[:, -1], np.zeros(2))
        np.testing.assert_array_equal(next_obs[:, -1], np.ones(2))
        np.testing.assert_array_equal(
            next_obs[0, -5:-1],
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        np.testing.assert_array_equal(
            next_obs[1, -5:-1],
            np.array([0.0, 0.0, 1.0, 0.0]),
        )

    def test_policy_never_selects_a_masked_action(self):
        agent = QMIXMADDQNAgent(
            state_dim=3,
            action_dim=4,
            n_agents=1,
            batch_size=2,
            warmup_steps=2,
            hidden_dim=8,
            mixing_embed_dim=4,
            device=torch.device("cpu"),
        )
        for parameter in agent.q_net.parameters():
            parameter.data.zero_()
        agent.q_net.net[-1].bias.data.copy_(
            torch.tensor([10.0, 3.0, 2.0, 1.0])
        )
        only_action_one = np.array([False, True, False, False])

        agent.epsilon = 0.0
        self.assertEqual(agent.select_action(np.zeros(3), only_action_one), 1)
        agent.epsilon = 1.0
        selected = {
            agent.select_action(np.zeros(3), only_action_one)
            for _ in range(20)
        }
        self.assertEqual(selected, {1})

    def test_mixer_is_monotonic_in_each_agent_q(self):
        torch.manual_seed(3)
        mixer = QMixer(n_agents=2, global_state_dim=6, mixing_embed_dim=4)
        state = torch.randn(5, 6)
        baseline_q = torch.randn(5, 2)
        improved_q = baseline_q.clone()
        improved_q[:, 0] += 1.0

        baseline_total = mixer(baseline_q, state)
        improved_total = mixer(improved_q, state)

        self.assertTrue(torch.all(improved_total >= baseline_total - 1e-6))

    def test_warmup_and_learning_interval(self):
        agent = QMIXMADDQNAgent(
            state_dim=3,
            action_dim=2,
            n_agents=1,
            batch_size=2,
            warmup_steps=4,
            learn_every=2,
            target_tau=0.01,
            hidden_dim=8,
            mixing_embed_dim=4,
            device=torch.device("cpu"),
        )
        obs = np.zeros((1, 3), dtype=np.float32)
        masks = np.ones((1, 2), dtype=bool)

        for _ in range(3):
            agent.store_joint(obs, [0], 0.0, obs, False, masks)
            self.assertIsNone(agent.learn())
        self.assertEqual(agent.learn_step_count, 0)

        agent.store_joint(obs, [0], 0.0, obs, True, masks)
        self.assertIsNotNone(agent.learn())
        self.assertEqual(agent.learn_step_count, 1)


if __name__ == "__main__":
    unittest.main()
