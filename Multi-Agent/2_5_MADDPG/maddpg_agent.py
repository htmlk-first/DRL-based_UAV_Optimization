"""MADDPG agent for cooperative multi-UAV continuous control."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    """Decentralized actor: local observation -> continuous action."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        final = self.net[-2]
        nn.init.uniform_(final.weight, -3e-3, 3e-3)
        nn.init.uniform_(final.bias, -3e-3, 3e-3)

    def forward(self, obs):
        return self.net(obs)


class CriticNetwork(nn.Module):
    """Centralized critic: joint observation + joint action -> Q-value."""

    def __init__(self, global_obs_dim, joint_action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(global_obs_dim + joint_action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, global_obs, joint_actions):
        x = torch.cat([global_obs, joint_actions], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x).squeeze(1)


class OUNoise:
    """Ornstein-Uhlenbeck noise for temporally correlated exploration."""

    def __init__(self, shape, mu=0.0, theta=0.15, sigma=0.2):
        self.shape = tuple(shape)
        self.mu = np.full(self.shape, mu, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(*self.shape).astype(np.float32)
        self.state = self.state + dx
        return self.state


class JointReplayBuffer:
    """Stores one joint transition per environment step."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs_batch,
        actions,
        rewards,
        next_obs_batch,
        done,
        guidance_actions,
    ):
        self.buffer.append((
            np.array(obs_batch, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs_batch, dtype=np.float32),
            float(done),
            np.array(guidance_actions, dtype=np.float32),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, guidance_actions = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(guidance_actions, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    """Multi-agent DDPG with centralized critics and decentralized actors."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        n_agents,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=128,
        buffer_capacity=200000,
        hidden_dim=256,
        noise_sigma=0.3,
        noise_sigma_min=0.02,
        noise_decay=0.999,
        grad_clip=1.0,
        reward_scale=0.01,
        warmup_steps=5000,
        learn_every=4,
        policy_delay=2,
        action_l2=1e-3,
        bc_weight=1.0,
        bc_weight_min=0.05,
        bc_decay_steps=100000,
        guidance_mix=0.5,
        guidance_decay_steps=50000,
        device=None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.global_obs_dim = obs_dim * n_agents
        self.joint_action_dim = action_dim * n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_sigma_min = noise_sigma_min
        self.noise_decay = noise_decay
        self.grad_clip = grad_clip
        self.reward_scale = reward_scale
        self.warmup_steps = max(batch_size, warmup_steps)
        self.learn_every = max(1, learn_every)
        self.policy_delay = max(1, policy_delay)
        self.action_l2 = action_l2
        self.bc_weight = bc_weight
        self.bc_weight_min = bc_weight_min
        self.bc_decay_steps = max(1, bc_decay_steps)
        self.guidance_mix = guidance_mix
        self.guidance_decay_steps = max(1, guidance_decay_steps)
        self.step_count = 0
        self.learn_step_count = 0

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.actors = nn.ModuleList([
            ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.target_actors = nn.ModuleList([
            ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.critics = nn.ModuleList([
            CriticNetwork(self.global_obs_dim, self.joint_action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.target_critics = nn.ModuleList([
            CriticNetwork(self.global_obs_dim, self.joint_action_dim, hidden_dim).to(self.device)
            for _ in range(n_agents)
        ])

        for target, source in zip(self.target_actors, self.actors):
            target.load_state_dict(source.state_dict())
            target.eval()
        for target, source in zip(self.target_critics, self.critics):
            target.load_state_dict(source.state_dict())
            target.eval()

        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=actor_lr, weight_decay=1e-6)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=critic_lr, weight_decay=1e-5)
            for critic in self.critics
        ]

        self.noise = OUNoise((n_agents, action_dim), sigma=noise_sigma)
        self.buffer = JointReplayBuffer(buffer_capacity)

    @staticmethod
    def _project_action_tensor(actions):
        norms = torch.linalg.vector_norm(actions, dim=-1, keepdim=True)
        return actions / norms.clamp(min=1.0)

    @staticmethod
    def project_actions(actions):
        actions = np.asarray(actions, dtype=np.float32)
        norms = np.linalg.norm(actions, axis=-1, keepdims=True)
        return actions / np.maximum(norms, 1.0)

    def select_actions(self, obs_batch, add_noise=True):
        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        actions = []

        for i, actor in enumerate(self.actors):
            actor.eval()
            with torch.no_grad():
                action = actor(obs_t[i].unsqueeze(0)).cpu().numpy()[0]
            actor.train()
            actions.append(action)

        actions = np.array(actions, dtype=np.float32)
        if add_noise:
            actions = actions + self.noise.sample()
        actions = np.clip(actions, -1.0, 1.0)
        return self.project_actions(actions).astype(np.float32)

    @property
    def current_guidance_mix(self):
        progress = min(self.step_count / self.guidance_decay_steps, 1.0)
        return self.guidance_mix * (1.0 - progress)

    @property
    def current_bc_weight(self):
        progress = min(
            self.learn_step_count / self.bc_decay_steps, 1.0
        )
        return max(
            self.bc_weight_min,
            self.bc_weight * (1.0 - progress),
        )

    def blend_guidance(self, actions, guidance_actions):
        mix = self.current_guidance_mix
        blended = (
            (1.0 - mix) * np.asarray(actions, dtype=np.float32)
            + mix * np.asarray(guidance_actions, dtype=np.float32)
        )
        return self.project_actions(blended).astype(np.float32)

    def store_joint(
        self,
        obs_batch,
        actions,
        rewards,
        next_obs_batch,
        done,
        guidance_actions=None,
    ):
        if guidance_actions is None:
            guidance_actions = np.zeros_like(actions, dtype=np.float32)
        self.buffer.push(
            obs_batch,
            actions,
            rewards,
            next_obs_batch,
            done,
            guidance_actions,
        )
        self.step_count += 1

    def learn(self):
        if len(self.buffer) < self.warmup_steps:
            return None, None
        if self.step_count % self.learn_every != 0:
            return None, None

        (
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            guidance_actions,
        ) = self.buffer.sample(self.batch_size)

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = (
            torch.FloatTensor(rewards).to(self.device) * self.reward_scale
        )
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        guidance_actions_t = torch.FloatTensor(
            guidance_actions
        ).to(self.device)

        batch_size = obs_t.shape[0]
        global_obs = obs_t.reshape(batch_size, self.global_obs_dim)
        next_global_obs = next_obs_t.reshape(batch_size, self.global_obs_dim)
        joint_actions = actions_t.reshape(batch_size, self.joint_action_dim)

        with torch.no_grad():
            next_actions = [
                self._project_action_tensor(
                    self.target_actors[i](next_obs_t[:, i, :])
                )
                for i in range(self.n_agents)
            ]
            next_joint_actions = torch.cat(next_actions, dim=1)

        critic_losses = []
        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = self.target_critics[i](next_global_obs, next_joint_actions)
                target = rewards_t[:, i] + self.gamma * target_q * (1 - dones_t)

            current_q = self.critics[i](global_obs, joint_actions)
            critic_loss = F.smooth_l1_loss(current_q, target)
            if not torch.isfinite(critic_loss):
                raise FloatingPointError(
                    "Non-finite MADDPG critic loss detected."
                )

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.grad_clip)
            self.critic_optimizers[i].step()
            critic_losses.append(float(critic_loss.item()))

        self.learn_step_count += 1
        actor_losses = []
        update_actors = self.learn_step_count % self.policy_delay == 0
        if update_actors:
            for critic in self.critics:
                for param in critic.parameters():
                    param.requires_grad_(False)

            for i in range(self.n_agents):
                policy_actions = []
                for j in range(self.n_agents):
                    action_j = self._project_action_tensor(
                        self.actors[j](obs_t[:, j, :])
                    )
                    if j != i:
                        action_j = action_j.detach()
                    policy_actions.append(action_j)
                policy_joint_actions = torch.cat(policy_actions, dim=1)
                action_penalty = policy_actions[i].pow(2).mean()
                bc_loss = F.mse_loss(
                    policy_actions[i],
                    guidance_actions_t[:, i, :],
                )
                actor_loss = (
                    -self.critics[i](global_obs, policy_joint_actions).mean()
                    + self.action_l2 * action_penalty
                    + self.current_bc_weight * bc_loss
                )
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError(
                        "Non-finite MADDPG actor loss detected."
                    )

                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actors[i].parameters(), self.grad_clip
                )
                self.actor_optimizers[i].step()
                actor_losses.append(float(actor_loss.item()))

            for critic in self.critics:
                for param in critic.parameters():
                    param.requires_grad_(True)

            for actor, target_actor in zip(self.actors, self.target_actors):
                self._soft_update(actor, target_actor)

        for critic, target_critic in zip(self.critics, self.target_critics):
            self._soft_update(critic, target_critic)

        mean_actor_loss = (
            float(np.mean(actor_losses)) if actor_losses else None
        )
        return float(np.mean(critic_losses)), mean_actor_loss

    def _soft_update(self, source, target):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data
            )

    def reset_noise(self):
        self.noise.reset()

    def decay_noise(self):
        self.noise.sigma = max(
            self.noise_sigma_min,
            self.noise.sigma * self.noise_decay,
        )

    def save(self, path):
        torch.save({
            "actors": [actor.state_dict() for actor in self.actors],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_critics": [critic.state_dict() for critic in self.target_critics],
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "noise_sigma": self.noise.sigma,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "n_agents": self.n_agents,
            "reward_scale": self.reward_scale,
            "warmup_steps": self.warmup_steps,
            "learn_every": self.learn_every,
            "policy_delay": self.policy_delay,
            "action_l2": self.action_l2,
            "bc_weight": self.bc_weight,
            "bc_weight_min": self.bc_weight_min,
            "bc_decay_steps": self.bc_decay_steps,
            "guidance_mix": self.guidance_mix,
            "guidance_decay_steps": self.guidance_decay_steps,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for actor, state in zip(self.actors, checkpoint["actors"]):
            actor.load_state_dict(state)
        for actor, state in zip(self.target_actors, checkpoint["target_actors"]):
            actor.load_state_dict(state)
        for critic, state in zip(self.critics, checkpoint["critics"]):
            critic.load_state_dict(state)
        for critic, state in zip(self.target_critics, checkpoint["target_critics"]):
            critic.load_state_dict(state)
        for optimizer, state in zip(self.actor_optimizers, checkpoint["actor_optimizers"]):
            optimizer.load_state_dict(state)
        for optimizer, state in zip(self.critic_optimizers, checkpoint["critic_optimizers"]):
            optimizer.load_state_dict(state)
        self.noise.sigma = checkpoint.get("noise_sigma", self.noise.sigma)
