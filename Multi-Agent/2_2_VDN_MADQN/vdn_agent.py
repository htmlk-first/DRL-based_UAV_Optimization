"""VDN-style CTDE agent for cooperative multi-UAV DQN."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
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


class JointReplayBuffer:
    """Stores one joint transition per environment step."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_batch, actions, team_reward, next_obs_batch, done):
        self.buffer.append((
            np.array(obs_batch, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            float(team_reward),
            np.array(next_obs_batch, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class VDNMADQNAgent:
    """Value-decomposition DQN with centralized training and decentralized execution."""

    def __init__(
        self,
        state_dim,
        action_dim,
        n_agents,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.997,
        batch_size=128,
        buffer_capacity=150000,
        target_update_freq=500,
        hidden_dim=256,
        grad_clip=5.0,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_net = AgentQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = AgentQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = JointReplayBuffer(buffer_capacity)
        self.learn_step_count = 0

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def select_actions(self, obs_batch):
        return np.array([self.select_action(obs) for obs in obs_batch], dtype=np.int64)

    def store_joint(self, obs_batch, actions, team_reward, next_obs_batch, done):
        self.buffer.push(obs_batch, actions, team_reward, next_obs_batch, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        batch_size = obs_t.shape[0]
        flat_obs = obs_t.reshape(batch_size * self.n_agents, self.state_dim)
        flat_next_obs = next_obs_t.reshape(batch_size * self.n_agents, self.state_dim)

        q_values = self.q_net(flat_obs).reshape(batch_size, self.n_agents, self.action_dim)
        chosen_q = q_values.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)
        q_total = chosen_q.sum(dim=1)

        with torch.no_grad():
            next_q_values = self.target_net(flat_next_obs).reshape(
                batch_size, self.n_agents, self.action_dim
            )
            next_agent_q = next_q_values.max(dim=2)[0]
            next_q_total = next_agent_q.sum(dim=1)
            target_total = rewards_t + self.gamma * next_q_total * (1 - dones_t)

        loss = self.loss_fn(q_total, target_total)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.learn_step_count += 1
        if self.learn_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_agents": self.n_agents,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)

