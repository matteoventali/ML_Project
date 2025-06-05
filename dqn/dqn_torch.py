import os
import numpy as np
import gymnasium as gym
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class QLearner:
    def __init__(self, env, max_episodes=3000, gamma=0.99, alpha=0.1, end_eps=0.01, start_eps=1.0, eps_decay=0.995):
        self.env = env
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.eps = start_eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay
        self.batch_size = 64
        self.buffer = ReplayBuffer(10000)

        self.q_network = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def _select_action(self, state):
        if random.random() < self.eps:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def _normalize(self, state):
        amplitudes = [2.5, 2.5, 10., 10., 6.2831855, 10., 1., 1.]
        return np.array(state) / np.array(amplitudes)

    def train_step(self):
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        current_q = self.q_network(states).gather(1, actions)
        next_q = self.q_network(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        total_rewards = []
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = self._normalize(state)
            done = truncated = False
            episode_reward = 0

            while not (done or truncated):
                action = self._select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = self._normalize(next_state)

                self.buffer.add(state, action, reward, next_state, done or truncated)
                state = next_state
                episode_reward += reward

                if len(self.buffer) > self.batch_size:
                    self.train_step()

            self.eps = max(self.end_eps, self.eps * self.eps_decay)
            total_rewards.append(episode_reward)
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Epsilon = {self.eps:.3f}")

        return total_rewards


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    ql = QLearner(env)
    ql.train()
