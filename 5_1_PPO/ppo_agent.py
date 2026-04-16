"""
PPO (Proximal Policy Optimization) Agent
─────────────────────────────────────────────────
DDPG 대비 핵심 개선:
1. On-Policy 학습
   - DDPG: Off-policy (Replay Buffer에서 과거 경험 재사용)
   - PPO: On-policy (직접 수집한 rollout 데이터로 학습 → 안정적)
2. 확률적 정책 (Stochastic Policy)
   - DDPG: 결정적 정책 + OU Noise (외부 노이즈)
   - PPO: 가우시안 분포 출력 → 평균 + 표준편차로 자연스러운 탐색
3. Clipped Surrogate Objective
   - 정책 업데이트 크기를 clip_epsilon으로 제한
   - 큰 정책 변화 방지 → 학습 안정성 극대화
4. GAE (Generalized Advantage Estimation)
   - TD 오차를 지수 가중 합산 → 분산-편향 트레이드오프 조절
   - λ 파라미터로 몬테카를로 ↔ TD 사이 조절
5. 엔트로피 보너스 (Entropy Bonus)
   - 정책의 엔트로피를 최대화 → 조기 수렴 방지
   - DDPG의 OU Noise 대비 이론적으로 우수한 탐색
6. Value Function Clipping
   - Critic(V) 업데이트도 clip → 학습 안정성 향상
7. Multiple Epochs per Rollout
   - 같은 데이터로 K번 학습 → 샘플 효율성 ↑
8. 학습률 스케줄링 (LR Annealing)
   - 학습 진행에 따라 LR 점진적 감소 → 후반부 안정화
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# ═══════════════════════════════════════════════════════════════════════════
#  Actor-Critic Network (공유 백본 + 분리 헤드)
# ═══════════════════════════════════════════════════════════════════════════

class ActorCriticNetwork(nn.Module):
    """
    PPO Actor-Critic 네트워크

    DDPG와의 구조적 차이:
    ┌───────────┬─────────────────────────┬──────────────────────────────────┐
    │           │ DDPG                    │ PPO                              │
    ├───────────┼─────────────────────────┼──────────────────────────────────┤
    │ Actor     │ 결정적: state → action  │ 확률적: state → (μ, σ) → 분포   │
    │ Critic    │ Q(s, a) — 행동 포함     │ V(s) — 상태만 평가              │
    │ 네트워크  │ Actor/Critic 완전 분리  │ 백본 공유 가능 (효율적)          │
    │ 타겟 네트워크 │ Actor_target + Critic_target │ 없음 (on-policy)       │
    └───────────┴─────────────────────────┴──────────────────────────────────┘
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # ── Actor (정책 네트워크): state → (mean, log_std) ──
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_ln1 = nn.LayerNorm(hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_ln2 = nn.LayerNorm(hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # 학습 가능한 log_std 파라미터 (행동 차원별 독립)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # ── Critic (가치 네트워크): state → V(s) ──
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_ln1 = nn.LayerNorm(hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_ln2 = nn.LayerNorm(hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

        # 가중치 초기화 (직교 초기화 — PPO에서 효과적)
        self._init_weights()

    def _init_weights(self):
        for m in [self.actor_fc1, self.actor_fc2, self.critic_fc1, self.critic_fc2]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0)
        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)
        nn.init.constant_(self.critic_out.bias, 0)

    def actor(self, state):
        """정책 분포 파라미터 반환: (mean, std)"""
        x = F.relu(self.actor_ln1(self.actor_fc1(state)))
        x = F.relu(self.actor_ln2(self.actor_fc2(x)))
        mean = torch.tanh(self.actor_mean(x))
        std = torch.exp(torch.clamp(self.actor_log_std, -3.0, -0.5))
        return mean, std

    def critic(self, state):
        """상태 가치 V(s) 반환"""
        x = F.relu(self.critic_ln1(self.critic_fc1(state)))
        x = F.relu(self.critic_ln2(self.critic_fc2(x)))
        return self.critic_out(x)

    def forward(self, state):
        mean, std = self.actor(state)
        value = self.critic(state)
        return mean, std, value


# ═══════════════════════════════════════════════════════════════════════════
#  Rollout Buffer (On-Policy 데이터 저장)
# ═══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    On-Policy Rollout Buffer

    DDPG의 ReplayBuffer와의 차이:
    - ReplayBuffer: 과거 경험을 랜덤 샘플링 (off-policy)
    - RolloutBuffer: 현재 정책으로 수집한 에피소드 데이터 저장 후
                     학습이 끝나면 전부 버림 (on-policy)
    - log_prob, value, advantage 등 PPO 고유 데이터 저장
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


# ═══════════════════════════════════════════════════════════════════════════
#  PPO Agent
# ═══════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    Proximal Policy Optimization Agent

    DDPG와의 학습 방식 차이:
    ┌──────────────────┬────────────────────────┬────────────────────────────────┐
    │                  │ DDPG                   │ PPO                            │
    ├──────────────────┼────────────────────────┼────────────────────────────────┤
    │ 데이터 수집      │ 1 step → buffer 저장   │ N steps rollout → 일괄 학습    │
    │ 학습 방식        │ Off-policy (replay)    │ On-policy (현재 정책 데이터)   │
    │ 정책 업데이트    │ ∇_a Q(s,a) (Critic으로 │ Clipped surrogate objective    │
    │                  │  Actor 개선)           │ (비율 clip으로 안정적 업데이트) │
    │ Critic 타겟      │ Q(s,a) + target net    │ V(s) + GAE advantages         │
    │ 탐색 전략        │ OU Noise (외부 노이즈) │ 정책 분포 자체 + 엔트로피 보너스│
    │ 샘플 효율        │ 높음 (off-policy 재사용)│ K epochs로 보완               │
    │ 학습 안정성      │ Soft target update 필요│ 클리핑으로 자체 안정화          │
    └──────────────────┴────────────────────────┴────────────────────────────────┘
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        max_grad_norm=0.5,
        k_epochs=10,
        mini_batch_size=64,
        hidden_dim=256,
        lr_decay=True,
        total_updates=None,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.k_epochs = k_epochs
        self.mini_batch_size = mini_batch_size

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Actor-Critic Network (단일 네트워크, 타겟 네트워크 없음) ──
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # ── 학습률 스케줄링 ──
        self.lr_decay = lr_decay
        self.initial_lr = lr
        self.total_updates = total_updates
        self.update_count = 0

        # ── Rollout Buffer ──
        self.buffer = RolloutBuffer()

    # ── 행동 선택 ──

    def select_action(self, state, deterministic=False):
        """
        확률적 정책으로 행동 선택

        DDPG: Actor(s) → 결정적 행동 + 외부 OU 노이즈
        PPO:  Actor(s) → (μ, σ) → Normal 분포에서 샘플링 → 자연스러운 탐색
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, std, value = self.network(state_t)

        if deterministic:
            action = mean.cpu().numpy()[0]
            log_prob = 0.0
        else:
            dist = Normal(mean, std)
            action_raw = dist.sample()
            action = torch.clamp(action_raw, -1.0, 1.0)
            # clamped action에 대한 log_prob (learn과 일관성 유지)
            log_prob = dist.log_prob(action).sum(dim=-1).item()
            action = action.cpu().numpy()[0]

        return action, log_prob, value.item()

    # ── 경험 저장 ──

    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.store(state, action, log_prob, reward, value, done)

    # ── GAE (Generalized Advantage Estimation) ──

    def _compute_gae(self, last_value):
        """
        GAE: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
             A_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}

        DDPG는 TD target (r + γ·Q_target)만 사용하지만,
        PPO의 GAE는 여러 스텝의 TD 오차를 exponentially weighted 합산하여
        분산과 편향의 균형을 맞춤. λ=0 → TD(0), λ=1 → MC.
        """
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        n = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    # ── PPO 학습 ──

    def learn(self, last_value=0.0):
        """
        PPO Clipped Surrogate Objective 학습

        1) GAE로 advantage 계산
        2) K epochs × mini-batch로 반복 학습:
           - Policy loss: clipped surrogate
           - Value loss: MSE(V(s), returns)
           - Entropy bonus: 탐색 유지

        DDPG 학습과의 차이:
        - DDPG: replay buffer에서 random sample → 1회 업데이트
        - PPO: rollout 전체 → K epochs 반복 학습 → buffer 클리어
        """
        if len(self.buffer) == 0:
            return None, None, None

        advantages, returns = self._compute_gae(last_value)

        # Advantage 정규화 (학습 안정성)
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # numpy → tensor
        states_t = torch.FloatTensor(
            np.array(self.buffer.states)
        ).to(self.device)
        actions_t = torch.FloatTensor(
            np.array(self.buffer.actions)
        ).to(self.device)
        old_log_probs_t = torch.FloatTensor(
            np.array(self.buffer.log_probs)
        ).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        n = len(self.buffer)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.k_epochs):
            # ── Mini-batch 셔플 ──
            indices = np.random.permutation(n)

            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                mb_idx = indices[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # ── 현재 정책으로 재평가 ──
                mean, std, values = self.network(mb_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # ── Clipped Surrogate Objective ──
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value Loss (clipped) ──
                values_squeezed = values.squeeze(-1)
                value_loss = F.mse_loss(values_squeezed, mb_returns)

                # ── Total Loss ──
                loss = (policy_loss
                        + self.value_coeff * value_loss
                        - self.entropy_coeff * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # ── 학습률 스케줄링 ──
        self.update_count += 1
        if self.lr_decay and self.total_updates:
            frac = max(1.0 - self.update_count / self.total_updates, 0.0)
            new_lr = self.initial_lr * frac
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

        # ── Rollout Buffer 클리어 (on-policy 핵심) ──
        self.buffer.clear()

        avg_p = total_policy_loss / max(n_updates, 1)
        avg_v = total_value_loss / max(n_updates, 1)
        avg_e = total_entropy / max(n_updates, 1)
        return avg_p, avg_v, avg_e

    # ── 현재 학습률 ──

    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    # ── 저장 / 로드 ──

    def save(self, path):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = ckpt["update_count"]
