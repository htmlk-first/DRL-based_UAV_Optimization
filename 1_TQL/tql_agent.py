"""
Tabular Q-Learning Agent
- Q-테이블 기반 강화학습
- epsilon-greedy 탐색 전략
"""
import numpy as np
import pickle
from collections import defaultdict


class TQLAgent:
    def __init__(self, n_actions, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-테이블: defaultdict로 새로운 상태에 대해 자동 초기화
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        """epsilon-greedy 행동 선택"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Q-Learning 업데이트: Q(s,a) <- Q(s,a) + lr * [r + γ*max(Q(s',·)) - Q(s,a)]"""
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data["q_table"])
        self.epsilon = data["epsilon"]
