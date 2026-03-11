import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from replay_buffer import ReplayBuffer


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):

        return self.net(x)


class DQNAgent:

    def __init__(self, state_dim, action_dim):

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        self.buffer = ReplayBuffer()

    def select_action(self, state, valid_actions):

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        q = self.policy_net(s).detach().cpu().numpy()[0]

        masked = np.full_like(q, -1e9)

        for a in valid_actions:
            masked[a] = q[a]

        return int(np.argmax(masked))

    def train(self):

        if len(self.buffer) < self.batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q = self.policy_net(s).gather(1, a)

        next_q = self.target_net(s2).max(1)[0].detach().unsqueeze(1)

        target = r + (1 - d) * self.gamma * next_q

        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):

        self.target_net.load_state_dict(self.policy_net.state_dict())