"""
Double DQN (DDQN) Agent
- Double Q-Learning: 온라인 네트워크가 행동 선택, 타겟 네트워크가 Q값 평가
  → Q값 과대추정(overestimation) 문제 해소
- Experience Replay
- Target Network (hard update)
- Gradient Clipping → 학습 안정화
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    """Q-Value를 근사하는 신경망"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
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


class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    """
    Double DQN Agent

    핵심 차이점 (DQN vs DDQN):
    - DQN:  target = r + γ * max_a' Q_target(s', a')
    - DDQN: target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
      → 온라인 네트워크로 최적 행동을 선택하고, 타겟 네트워크로 Q값을 평가
      → max 연산자의 긍정적 편향(positive bias) 제거
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.998,
        batch_size=64,
        buffer_capacity=50000,
        target_update_freq=200,
        hidden_dim=128,
        grad_clip=10.0,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
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

        # 네트워크
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_capacity)
        self.learn_step_count = 0

    def select_action(self, state):
        """epsilon-greedy 행동 선택"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Double DQN 미니배치 학습

        DQN:  next_q = max_a' Q_target(s', a')
        DDQN: next_q = Q_target(s', argmax_a' Q_online(s', a'))
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 현재 Q값
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ── Double DQN Target ──
        with torch.no_grad():
            # 1) 온라인 네트워크로 다음 상태의 최적 행동 선택
            best_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)

            # 2) 타겟 네트워크로 해당 행동의 Q값 평가
            next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)

            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (학습 안정화)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)

        self.optimizer.step()

        # Target 네트워크 업데이트
        self.learn_step_count += 1
        if self.learn_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
