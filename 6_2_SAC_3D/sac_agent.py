"""
SAC (Soft Actor-Critic) Agent — 3D 확장
─────────────────────────────────────────────────
6_1_SAC와 동일한 SAC 알고리즘:
1. Off-Policy 학습 (Replay Buffer)
2. Twin Q-Networks (Clipped Double-Q)
3. Gaussian Policy with Reparameterization Trick (Tanh Squashing)
4. Automatic Entropy Tuning (α)
5. Soft Target Updates (τ)

3D 확장 요약:
- action_dim: 2 → 3 (dx, dy, dz)
- state_dim: 16 → 24 (3D 좌표 + 14방향 감지 + 3D 방향)
- 네트워크 구조/하이퍼파라미터는 동일 (자동 적응)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random


# ──────────────────────────────────────────────────
#  Replay Buffer (Off-Policy)
# ──────────────────────────────────────────────────

class ReplayBuffer:
    """경험 재생 버퍼"""

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────
#  Network Definitions
# ──────────────────────────────────────────────────

def _mlp(in_dim, hidden_dim, out_dim, num_hidden=2):
    """간단한 MLP 빌더"""
    layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    for _ in range(num_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """
    Squashed Gaussian Policy (Tanh):
        a = tanh(μ + σ * ε),   ε ~ N(0,1)
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """Reparameterization trick + squashing"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()                         # reparameterized sample
        action = torch.tanh(x_t)                      # squash to [-1, 1]

        # log_prob with correction for tanh squashing
        log_prob = dist.log_prob(x_t)                 # (B, action_dim)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)

        return action, log_prob

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class TwinQNetwork(nn.Module):
    """Twin Q-Networks: Q1(s,a), Q2(s,a)"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, hidden_dim, 1, num_hidden=2)
        self.q2 = _mlp(state_dim + action_dim, hidden_dim, 1, num_hidden=2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ──────────────────────────────────────────────────
#  SAC Agent
# ──────────────────────────────────────────────────

class SACAgent:
    """
    Soft Actor-Critic with:
    - Automatic entropy coefficient (alpha) tuning
    - Twin Q-network (clipped double-Q)
    - Soft target network updates
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100_000,
        batch_size=256,
        initial_alpha=0.2,
        device=None,
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # ── Networks ──
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Optimizers ──
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # ── Automatic Entropy Tuning ──
        self.target_entropy = -0.5 * float(action_dim)
        self.log_alpha = torch.tensor(
            np.log(initial_alpha), dtype=torch.float32,
            device=self.device, requires_grad=True,
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # ── Replay Buffer ──
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # ── Logging ──
        self.actor_loss_log = []
        self.critic_loss_log = []
        self.alpha_log = []

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    # ── Action Selection ──

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                a = self.actor.deterministic(s)
            else:
                a, _ = self.actor.sample(s)
        return a.cpu().numpy().flatten()

    # ── Update ──

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        alpha = self.log_alpha.exp().detach()

        # ── 1) Critic Update ──
        with torch.no_grad():
            next_a, next_log_prob = self.actor.sample(s2)
            q1_target, q2_target = self.critic_target(s2, next_a)
            q_target = torch.min(q1_target, q2_target)
            y = r + (1.0 - d) * self.gamma * (q_target - alpha * next_log_prob)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ── 2) Actor Update ──
        new_a, new_log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        actor_loss = (alpha * new_log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ── 3) Alpha (Entropy) Update ──
        alpha_loss = -(
            self.log_alpha * (new_log_prob.detach() + self.target_entropy)
        ).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ── 4) Soft Target Update ──
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)

        # ── Logging ──
        self.actor_loss_log.append(actor_loss.item())
        self.critic_loss_log.append(critic_loss.item())
        self.alpha_log.append(self.alpha)

    # ── Save / Load ──

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_optim.defaults["lr"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
