"""
Deep Recurrent Q Network (DQN + LSTM)
Author: zenggo
Date: 2020.6
"""
import numpy as np
from common.agent import Agent
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F


class Agent_DRQN(Agent):

    def __init__(self, n_obs, n_act, max_mem_size=300, lr=1e-3, epsilon=0.999, gamma=0.9, model_params={}):
        super(Agent_DRQN, self).__init__(n_obs, n_act)
        self.drqn = DRQN_model(n_obs, n_act, **model_params)
        # internal memory
        self.last_states = None
        # training
        self.is_training = False
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.drqn.parameters(), lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.training_count = 0
        # replay memory
        self.replay = ReplayMemory(max_mem_size)

    @property
    def decayed_epsilon(self):
        e = np.power(self.epsilon, self.training_count)
        return np.max([e, 0.05])

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def reset(self):
        self.last_states = None

    def respond(self, obs):
        x = torch.tensor([obs])
        with torch.no_grad():
            Q, self.last_states = self.drqn(x, self.last_states)
            action = Q.argmax().item()
        if self.is_training and np.random.uniform() < self.decayed_epsilon:
            action = np.random.randint(0, self.n_act)
        return action

    def learn(self, obs, next_obs, action, reward, done, target_act=None):
        self.replay.store_transition(obs, action, reward)
        if done:
            if self.replay.is_available:
                self._train()
                self.training_count += 1
            self.replay.start_new_episode()

    def _train(self):
        trajectory = self.replay.sample()
        Q_est_list = None
        Q_tgt_list = None
        _hstates = None
        Q_table_list = []

        for obs, _, _ in trajectory:
            x = torch.tensor([obs])
            Q, _hstates = self.drqn(x, _hstates)
            Q_table_list.append(Q)

        ep_len = len(trajectory)
        for i in range(ep_len):
            _, action, reward = trajectory[i]

            Q_est = Q_table_list[i][0, action]
            if Q_est_list is None:
                Q_est_list = Q_est.reshape(1,1)
            else:
                Q_est_list = torch.cat([Q_est_list, Q_est.reshape(1,1)])

            if i == ep_len-1:
                max_next_Q = torch.zeros(1, requires_grad=False)
            else:
                max_next_Q = Q_table_list[i+1].max().clone().detach()
            Q_tgt = reward + self.gamma * max_next_Q
            if Q_tgt_list is None:
                Q_tgt_list = Q_tgt.reshape(1,1)
            else:
                Q_tgt_list = torch.cat([Q_tgt_list, Q_tgt.reshape(1,1)])

        loss = self.criterion(Q_est_list, Q_tgt_list)
        loss.backward(create_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()



class ReplayMemory:
    def __init__(self, max_mem_size):
        self.max_mem_size = max_mem_size
        self.memory = []
        self.mem_count = 0
        self.memory.append([])

    @property
    def current_ep_trajectory(self):
        idx = self.mem_count % self.max_mem_size
        return self.memory[idx]

    def store_transition(self, state, action, reward):
        m = self.current_ep_trajectory
        m.append((state, action, reward))

    def start_new_episode(self):
        self.mem_count += 1
        if self.mem_count < self.max_mem_size:
            self.memory.append([])
        else:  # replace old memory
            idx = self.mem_count % self.max_mem_size
            self.memory[idx] = []

    @property
    def is_available(self):
        return self.mem_count > 0

    def sample(self):
        n_mem = len(self.memory)
        idx = np.random.randint(0, n_mem)
        return self.memory[idx]



class DRQN_model(nn.Module):
    def __init__(self, n_obs, n_actions, lstm_hidden_size=50, n_lstm_layers=1, linear_hidden_size=50, n_linear_layers=2):
        super(DRQN_model, self).__init__()
        self.embedding = nn.Embedding(n_obs, lstm_hidden_size)
        # LSTM layers as a internal memory
        self.lstm = nn.ModuleList([nn.LSTMCell(lstm_hidden_size, lstm_hidden_size) for _ in range(n_lstm_layers)])
        # Linear layers as Q network
        if n_linear_layers == 1:
            qnet = [nn.Linear(lstm_hidden_size, n_actions)]
        else:
            qnet = [nn.Linear(lstm_hidden_size, linear_hidden_size)]
            for i in np.arange(1, n_linear_layers-1):
                qnet.append(nn.Linear(linear_hidden_size, linear_hidden_size))
            qnet.append(nn.Linear(linear_hidden_size, n_actions))
        self.qnet = nn.ModuleList(qnet)

    def forward(self, x, last_states=None):
        n_lstm = len(self.lstm)
        if last_states is not None:
            assert len(last_states) == n_lstm
        else:
            last_states = [None] * n_lstm
        states = []
        hin = self.embedding(x)

        for n in range(n_lstm):
            ht, ct = self.lstm[n](hin, last_states[n])
            states.append((ht, ct))
            hin = ht

        h = states[-1][0]
        for i in range(len(self.qnet)-1):
            h = self.qnet[i](h)
            h = F.relu(h)
        Q = self.qnet[-1](h)

        return Q, states