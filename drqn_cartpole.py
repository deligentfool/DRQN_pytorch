import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym


class drqn_net(nn.Module):
    def __init__(self, observation_dim, action_dim, time_step=1, layer_num=1, hidden_num=128):
        super(drqn_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.time_step = time_step
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.lstm = nn.LSTM(self.observation_dim, self.hidden_num, self.layer_num, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_num, 128)
        # nn.LSTM(input_size, hidden_num, layer_num, batch_first=True)
        self.fc2 = nn.Linear(128, self.action_dim)

    def forward(self, observation, hidden=None):
        if not hidden:
            h0 = torch.zeros([self.layer_num, observation.size(0), self.hidden_num])
            c0 = torch.zeros([self.layer_num, observation.size(0), self.hidden_num])
            hidden = (h0, c0)
            # hidden [layer_num, batch_size, hidden_num]
        # lstm-x-input [batch_size, time_step, input_size]
        # lstm-x-output [batch_size, time_step, input_size]
        x, new_hidden = self.lstm(observation, hidden)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x, new_hidden

    def act(self, observation, epsilon, hidden):
        q_values, new_hidden = self.forward(observation, hidden)
        if random.random() > epsilon:
            action = q_values.max(-1)[1].detach()[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action, new_hidden


class recurrent_replay_buffer(object):
    def __init__(self, capacity, seq_len=50):
        self.capacity = capacity
        self.seq_len = seq_len
        self.memory = deque(maxlen=self.capacity)
        self.memory.append([])

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        self.memory[-1].append([observation, action, reward, next_observation, done])

        if len(self.memory[-1]) == self.seq_len:
            self.memory.append([])

    def sample(self, batch_size=32):
        observation = []
        action = []
        reward = []
        next_observation = []
        done = []
        batch = random.sample(list(self.memory)[: -1], batch_size)
        for i in range(batch_size):
            obs, act, rew, next_obs, don = zip(* batch[i])
            obs = np.expand_dims(np.concatenate(obs, 0), 0)
            next_obs = np.expand_dims(np.concatenate(next_obs, 0), 0)
            observation.append(obs)
            action.append(act)
            reward.append(rew)
            next_observation.append(next_obs)
            done.append(don)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


def train(buffer, target_model, eval_model, gamma, optimizer, loss_fn, count, soft_update_freq, batch_size):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    q_values, _ = eval_model.forward(observation)
    next_q_values, _ = target_model.forward(next_observation)
    argmax_actions = eval_model.forward(next_observation)[0].max(-1)[1].detach()
    next_q_value = next_q_values.gather(-1, argmax_actions.unsqueeze(-1)).squeeze(-1)
    q_value = q_values.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    expected_q_value = reward + gamma * (1 - done) * next_q_value

    #loss = loss_fn(q_value, expected_q_value.detach())
    loss = (expected_q_value.detach() - q_value).pow(2)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % soft_update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 1e-3
    soft_update_freq = 100
    capacity = 10000
    exploration = 300
    epsilon_init = 0.9
    epsilon_min = 0.05
    decay = 0.99
    episode = 1000000
    render = False
    seq_len = 50
    batch_size = 8

    env = gym.make('CartPole-v0')
    env = env.unwrapped
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    target_net = drqn_net(observation_dim, action_dim)
    eval_net = drqn_net(observation_dim, action_dim)
    eval_net.load_state_dict(target_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = recurrent_replay_buffer(capacity, seq_len)
    loss_fn = nn.MSELoss()
    epsilon = epsilon_init
    count = 0

    weight_reward = None
    for i in range(episode):
        obs = env.reset()
        hidden = None
        if epsilon > epsilon_min:
            epsilon = epsilon * decay
        reward_total = 0
        if render:
            env.render()
        while True:
            action, hidden = eval_net.act(torch.FloatTensor(np.expand_dims(np.expand_dims(obs, 0), 0)), epsilon, hidden)
            count += 1
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            if render:
                env.render()
            if i > exploration:
                train(buffer, target_net, eval_net, gamma, optimizer, loss_fn, count, soft_update_freq, batch_size)

            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break

