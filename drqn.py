import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import deque
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class drqn_net(nn.Module):
    def __init__(self, action_dim, hidden_dim, layer_num=1):
        super(drqn_net, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.conv1 = nn.Conv2d(3, 32, 10, 5)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 64, 2, 1)
        self.lstm = nn.LSTM(19 * 14 * 64,
                            self.hidden_dim,
                            self.layer_num,
                            batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x, hidden=None):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.flatten(out, 1, -1)
        if not hidden:
            h0 = torch.zeros(self.layer_num, out.size(0),
                             self.hidden_dim).to(device=device)
            c0 = torch.zeros(self.layer_num, out.size(0),
                             self.hidden_dim).to(device=device)
            hidden = (h0, c0)
        out = out.unsqueeze(dim=1)
        out, new_hidden = self.lstm(out, hidden)
        out = self.fc1(out[:, -1, :])
        out = self.fc2(F.relu(out))
        return out, new_hidden


class replay_memory(object):
    def __init__(self, memory_num=100):
        self.memory_num = memory_num
        self.memory = deque(maxlen=self.memory_num)

    def create_memory(self):
        self.memory.append([])

    def store(self, observation, action, reward):
        self.memory[-1].append([observation, action, reward])

    def sample(self):
        sample_index = random.randint(0, len(self.memory) - 1)
        return self.memory[sample_index]

    def size(self):
        return len(self.memory)


class implement(object):
    def __init__(self,
                 hidden_dim,
                 gamma=0.9,
                 learning_rate=1e-3,
                 epsilon=0.9,
                 decay=1e-2,
                 min_epsilon=0.1,
                 memory_num=100,
                 max_iter=100000,
                 render=False):
        self.env = gym.make('Frostbite-v0')
        self.env = self.env.unwrapped
        self.action_dim = self.env.action_space.n
        self.hidden_dim = hidden_dim
        self.episode = 0
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.render = render
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.memory_num = memory_num
        self.max_iter = max_iter
        self.net = drqn_net(self.action_dim, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.learning_rate)
        self.buffer = replay_memory(self.memory_num)

    def img_to_tensor(self, img):
        img_tensor = torch.FloatTensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def img_list_to_batch(self, img_list):
        temp_tensor = self.img_to_tensor(img_list[0])
        temp_tensor = temp_tensor.unsqueeze(dim=0)
        for i in range(1, len(img_list)):
            temp_tensor_ = self.img_to_tensor(img_list[i])
            temp_tensor_ = temp_tensor_.unsqueeze(dim=0)
            temp_tensor = torch.cat([temp_tensor, temp_tensor_], dim=0)
        return temp_tensor

    def train(self):
        memo = self.buffer.sample()
        observation_list = []
        action_list = []
        reward_list = []
        for i in range(len(memo)):
            observation_list.append(memo[i][0])
            action_list.append(memo[i][1])
            reward_list.append(memo[i][2])
        Q, _ = self.net.forward(self.img_list_to_batch(observation_list))
        Q_est = torch.clone(Q)
        for t in range(len(memo) - 1):
            max_next_Q = torch.max(Q_est[t + 1, :]).clone().detach()
            Q_est[t, action_list[t]] = reward_list[t] + self.gamma * max_next_Q
        Q_est[-1, action_list[-1]] = reward_list[-1]
        loss = self.loss_fn(Q, Q_est)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, observation, hidden):
        q, new_hidden = self.net.forward(self.img_list_to_batch([observation]),
                                         hidden)
        if random.random() < self.epsilon:
            action = torch.argmax(q.squeeze()).data.item()
        else:
            action = random.randint(0, self.action_dim - 1)
        return action, new_hidden

    def run(self):
        random.seed()
        iter_count = 0
        for episode in range(self.max_iter):
            observation = self.env.reset()
            new_hidden = None
            if self.render:
                self.env.render()
            self.epsilon = self.epsilon * (1 - self.decay)
            self.buffer.create_memory()
            reward_total = 0
            while True:
                action, new_hidden = self.select_action(
                    observation, new_hidden)
                next_observation, reward, done, info = self.env.step(action)
                self.buffer.store(observation, action, reward)
                observation = next_observation
                reward_total += reward
                if self.render:
                    self.env.render()
                if done:
                    print("episode: {} reward: {} epsilon: {:.3f}".format(
                        episode + 1, reward_total, self.epsilon))
                    break
            if episode >= 20:
                self.train()


if __name__ == '__main__':
    implement_test = implement(hidden_dim=32, render=False)
    implement_test.run()