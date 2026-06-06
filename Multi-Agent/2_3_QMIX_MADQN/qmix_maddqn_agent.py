"""QMIX-MADDQN agent for cooperative multi-UAV control."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AgentQNetwork(nn.Module):
    """Shared per-agent Q-network used by decentralized execution."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class QMixer(nn.Module):
    """Monotonic QMIX mixer conditioned on the flattened joint observation."""

    def __init__(self, n_agents, global_state_dim, mixing_embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.global_state_dim = global_state_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.max_mixing_weight = 2.0
        self.state_norm = nn.LayerNorm(global_state_dim)

        self.hyper_w1 = nn.Linear(global_state_dim, n_agents * mixing_embed_dim)
        self.hyper_b1 = nn.Linear(global_state_dim, mixing_embed_dim)
        self.hyper_w2 = nn.Linear(global_state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in (self.hyper_w1, self.hyper_b1, self.hyper_w2):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.constant_(module.bias, 0.0)
        for module in self.hyper_b2:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, agent_qs, global_states):
        batch_size = agent_qs.shape[0]
        global_states = self.state_norm(global_states)

        # Absolute hypernetwork outputs are the standard QMIX monotonic
        # parameterization. Unlike softplus(0) ~= 0.693, this also preserves
        # the deliberately small initialization from _init_weights().
        w1 = self.hyper_w1(global_states).abs().clamp(max=self.max_mixing_weight)
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(global_states).view(batch_size, 1, self.mixing_embed_dim)

        hidden = torch.bmm(agent_qs.view(batch_size, 1, self.n_agents), w1) + b1
        hidden = F.elu(hidden)

        w2 = self.hyper_w2(global_states).abs().clamp(max=self.max_mixing_weight)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(global_states).view(batch_size, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size)


class JointReplayBuffer:
    """Stores one joint transition per environment step."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs_batch,
        actions,
        team_reward,
        next_obs_batch,
        done,
        next_action_masks,
    ):
        self.buffer.append((
            np.array(obs_batch, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            float(team_reward),
            np.array(next_obs_batch, dtype=np.float32),
            float(done),
            np.array(next_action_masks, dtype=bool),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, next_action_masks = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_action_masks, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


class QMIXMADDQNAgent:
    """QMIX with Double-DQN targets and decentralized shared-Q execution."""

    def __init__(
        self,
        state_dim,
        action_dim,
        n_agents,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.997,
        batch_size=128,
        buffer_capacity=150000,
        target_update_freq=3000,
        hidden_dim=256,
        mixing_embed_dim=32,
        grad_clip=1.0,
        warmup_steps=10000,
        learn_every=4,
        target_tau=0.005,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.global_state_dim = state_dim * n_agents
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.warmup_steps = max(batch_size, warmup_steps)
        self.learn_every = max(1, learn_every)
        self.target_tau = target_tau

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_net = AgentQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = AgentQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.mixer = QMixer(n_agents, self.global_state_dim, mixing_embed_dim).to(self.device)
        self.target_mixer = QMixer(n_agents, self.global_state_dim, mixing_embed_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()),
            lr=lr,
            weight_decay=1e-5,
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = JointReplayBuffer(buffer_capacity)
        self.learn_step_count = 0
        self.learn_call_count = 0

    @staticmethod
    def _validated_action_mask(action_mask, action_dim):
        if action_mask is None:
            return np.ones(action_dim, dtype=bool)
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.shape != (action_dim,):
            raise ValueError(
                f"Expected action mask shape {(action_dim,)}, got {mask.shape}"
            )
        return mask if mask.any() else np.ones(action_dim, dtype=bool)

    @staticmethod
    def _mask_q_values(q_values, action_masks):
        valid = action_masks.any(dim=-1, keepdim=True)
        safe_masks = torch.where(valid, action_masks, torch.ones_like(action_masks))
        return q_values.masked_fill(~safe_masks, torch.finfo(q_values.dtype).min)

    def select_action(self, state, action_mask=None):
        action_mask = self._validated_action_mask(action_mask, self.action_dim)
        valid_actions = np.flatnonzero(action_mask)
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            mask_t = torch.as_tensor(
                action_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            q_values = self._mask_q_values(q_values, mask_t)
        return int(q_values.argmax(dim=1).item())

    def select_actions(self, obs_batch, action_masks=None):
        if action_masks is None:
            action_masks = [None] * len(obs_batch)
        if len(action_masks) != len(obs_batch):
            raise ValueError("action_masks length must match obs_batch")
        return np.array([
            self.select_action(obs, mask)
            for obs, mask in zip(obs_batch, action_masks)
        ], dtype=np.int64)

    def store_joint(
        self,
        obs_batch,
        actions,
        team_reward,
        next_obs_batch,
        done,
        next_action_masks=None,
    ):
        if next_action_masks is None:
            next_action_masks = np.ones(
                (self.n_agents, self.action_dim), dtype=bool
            )
        self.buffer.push(
            obs_batch,
            actions,
            team_reward,
            next_obs_batch,
            done,
            next_action_masks,
        )

    def learn(self):
        self.learn_call_count += 1
        if len(self.buffer) < self.warmup_steps:
            return None
        if self.learn_call_count % self.learn_every != 0:
            return None

        (
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            next_action_masks,
        ) = self.buffer.sample(self.batch_size)

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        next_action_masks_t = torch.as_tensor(
            next_action_masks, dtype=torch.bool, device=self.device
        )

        batch_size = obs_t.shape[0]
        global_states = obs_t.reshape(batch_size, self.global_state_dim)
        next_global_states = next_obs_t.reshape(batch_size, self.global_state_dim)
        flat_obs = obs_t.reshape(batch_size * self.n_agents, self.state_dim)
        flat_next_obs = next_obs_t.reshape(batch_size * self.n_agents, self.state_dim)

        q_values = self.q_net(flat_obs).reshape(batch_size, self.n_agents, self.action_dim)
        chosen_q = q_values.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)
        q_total = self.mixer(chosen_q, global_states)

        with torch.no_grad():
            next_online_q = self.q_net(flat_next_obs).reshape(
                batch_size, self.n_agents, self.action_dim
            )
            next_online_q = self._mask_q_values(
                next_online_q, next_action_masks_t
            )
            next_actions = next_online_q.argmax(dim=2, keepdim=True)

            next_target_q = self.target_net(flat_next_obs).reshape(
                batch_size, self.n_agents, self.action_dim
            )
            next_chosen_q = next_target_q.gather(2, next_actions).squeeze(-1)
            next_q_total = self.target_mixer(next_chosen_q, next_global_states)
            target_total = rewards_t + self.gamma * next_q_total * (1 - dones_t)

        loss = self.loss_fn(q_total, target_total)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite QMIX loss detected; stopping before corrupting the model."
            )

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                list(self.q_net.parameters()) + list(self.mixer.parameters()),
                self.grad_clip,
            )
        self.optimizer.step()

        self.learn_step_count += 1
        if self.target_tau > 0.0:
            self._soft_update_targets()
        elif self.learn_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return float(loss.item())

    def _soft_update_targets(self):
        with torch.no_grad():
            for target_param, param in zip(
                self.target_net.parameters(), self.q_net.parameters()
            ):
                target_param.data.lerp_(param.data, self.target_tau)
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.lerp_(param.data, self.target_tau)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_agents": self.n_agents,
            "global_state_dim": self.global_state_dim,
            "warmup_steps": self.warmup_steps,
            "learn_every": self.learn_every,
            "target_tau": self.target_tau,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
