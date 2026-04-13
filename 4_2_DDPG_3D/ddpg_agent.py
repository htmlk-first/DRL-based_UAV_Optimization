"""
DDPG (Deep Deterministic Policy Gradient) Agent for 3D UAV Environment
──────────────────────────────────────────────────────────────────────────
3_2_DDQN_3D 대비 핵심 개선:
1. Actor-Critic 구조
   - Actor  : 상태 → 3D 연속 행동 (정책 네트워크)
   - Critic : (상태, 행동) → Q값 (가치 네트워크)
2. 3D 연속 행동 공간 (Continuous Action Space)
   - tanh 출력으로 [-1, 1]^3 범위의 (dx, dy, dz) 생성
3. Soft Target Update (τ-기반 Polyak 평균)
   - DDQN의 hard update (주기적 전체 복사) 대비 안정적
4. Ornstein-Uhlenbeck (OU) Noise
   - 3차원 시간 상관 탐색 노이즈 → 부드러운 3D 경로 탐색
   - DDQN의 ε-greedy 대비 연속 공간에 적합
5. Layer Normalization → 학습 안정성 향상
6. Gradient Clipping (Actor + Critic 모두)
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
#  Networks
# ═══════════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """정책 네트워크: 상태 → 3D 연속 행동 (tanh ∈ [-1, 1]^3)"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 출력 레이어 가중치를 작게 초기화
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """가치 네트워크: (상태, 행동) → Q값"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# ═══════════════════════════════════════════════════════════════════════════
#  Ornstein-Uhlenbeck Noise (3D)
# ═══════════════════════════════════════════════════════════════════════════

class OUNoise:
    """
    Ornstein-Uhlenbeck 과정 — 3차원 시간 상관 탐색 노이즈

    3D 공간에서 연속적·방향성 있는 탐색을 위해
    이전 노이즈와 상관된 부드러운 3D 노이즈를 생성.
    """

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = (self.theta * (self.mu - self.state)
              + self.sigma * np.random.randn(len(self.mu)))
        self.state += dx
        return self.state


# ═══════════════════════════════════════════════════════════════════════════
#  Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Experience Replay Buffer (연속 행동 대응)"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════
#  DDPG Agent (3D)
# ═══════════════════════════════════════════════════════════════════════════

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent (3D)

    DDQN 3D와의 구조적 차이:
    ┌──────────────┬───────────────────────┬──────────────────────────────┐
    │              │ DDQN 3D               │ DDPG 3D                      │
    ├──────────────┼───────────────────────┼──────────────────────────────┤
    │ 행동 공간    │ Discrete(6)           │ Box(-1,1, shape=(3,))        │
    │ 네트워크     │ Q-Network × 2        │ Actor + Critic (× 2 target)  │
    │ 타겟 업데이트│ Hard copy (N step)    │ Soft update (τ-averaging)    │
    │ 탐색 전략    │ ε-greedy              │ OU Noise (3D 시간 상관)      │
    │ 학습         │ argmax Q → 행동 선택  │ ∇_a Q(s,a) → 정책 개선      │
    │ 이동 방향    │ 6방향 (격자)          │ 임의 방향 (연속 3D)          │
    └──────────────┴───────────────────────┴──────────────────────────────┘
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=128,
        buffer_capacity=200000,
        hidden_dim=256,
        noise_sigma=0.2,
        noise_sigma_min=0.01,
        noise_decay=0.9995,
        grad_clip=1.0,
        learn_every=2,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.noise_sigma_min = noise_sigma_min
        self.noise_decay = noise_decay
        self.learn_every = learn_every
        self.step_count = 0

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Actor ──
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        # ── Critic ──
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # ── Optimizers (Actor/Critic 분리 학습률) ──
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── OU Noise (3D) ──
        self.noise = OUNoise(action_dim, sigma=noise_sigma)

        # ── Replay Buffer ──
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── 행동 선택 ──

    def select_action(self, state, add_noise=True):
        """
        Actor 네트워크로 3D 행동 선택 + OU 노이즈

        DDQN 3D: ε 확률로 6방향 중 랜덤 선택 (불연속)
        DDPG 3D: 결정적 정책 + 3D OU 노이즈 → 부드러운 3D 탐색
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()

        if add_noise:
            noise = self.noise.sample()
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    # ── 경험 저장 ──

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

    # ── 학습 ──

    def learn(self):
        """
        DDPG 미니배치 학습 (3D 환경)

        Critic 업데이트:
          target = r + γ * Q_target(s', Actor_target(s'))
          loss   = MSE(Q(s, a), target)

        Actor 업데이트:
          loss = -mean( Q(s, Actor(s)) )    ← Q값을 최대화하는 방향으로 정책 개선

        Soft Target Update:
          θ_target ← τ·θ + (1-τ)·θ_target  (Actor & Critic 모두)
        """
        if len(self.buffer) < self.batch_size:
            return None, None
        if self.step_count % self.learn_every != 0:
            return None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ── Critic 업데이트 ──
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target = rewards_t + self.gamma * target_q * (1 - dones_t)

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # ── Actor 업데이트 ──
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ── Soft Target Update (τ-averaging) ──
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        """
        Soft Target Update: θ_target ← τ·θ_source + (1-τ)·θ_target

        DDQN 3D는 N 스텝마다 가중치를 통째로 복사(hard copy)하여
        학습이 불연속적으로 변하지만, DDPG의 soft update는
        매 스텝 미세하게 반영하여 학습 안정성을 높임.
        """
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(self.tau * src_p.data + (1 - self.tau) * tgt_p.data)

    # ── 노이즈 감쇠 ──

    def decay_noise(self):
        """OU 노이즈 σ를 감쇠 → 학습 후반 탐색 축소"""
        self.noise.sigma = max(
            self.noise_sigma_min,
            self.noise.sigma * self.noise_decay,
        )

    # ── 저장 / 로드 ──

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "noise_sigma": self.noise.sigma,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.noise.sigma = ckpt["noise_sigma"]
