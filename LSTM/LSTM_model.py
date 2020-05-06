import torch
import torch.nn as nn
from torch.optim import Adam
from common.agent import Agent
import os
import json


class Agent_LSTM(Agent):
    def __init__(self, n_obs, n_act, n_hidden, lr=0.01):
        super(Agent_LSTM, self).__init__(n_obs, n_act)
        self.n_hidden = n_hidden
        # one-hot features of the observations, fixed
        self.embedding = torch.eye(n_obs)
        # LSTM as a classifier | action predictor
        self.model = LSTModel(n_obs, n_hidden, n_act)
        # internal memory
        self.last_states = None
        self.last_output = None
        # training
        self.is_training = False
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.lr = lr

    def _obs2input(self, obs):
        return self.embedding[obs:obs+1]  # (1, n_obs)

    def train(self):
        self.model.train()
        self.is_training = True

    def eval(self):
        self.model.eval()
        self.is_training = False

    def reset(self):
        self.last_states = None

    def respond(self, obs):
        inputs = self._obs2input(obs)
        if self.is_training:
            output, self.last_states = self.model(inputs, self.last_states)
        else:
            with torch.no_grad():
                output, self.last_states = self.model(inputs, self.last_states)
        self.last_output = output
        action = output.argmax().item()
        return action

    def learn(self, obs, action, reward, done, target_act):
        loss = self.criterion(self.last_output, torch.tensor([target_act]))
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        params_path = os.path.join(dir, name)
        torch.save(self.model.state_dict(), params_path)
        configs_path = os.path.join(dir, name+'_configs.json')
        configs = {
            "n_obs": self.n_obs,
            "n_act": self.n_act,
            "n_hidden": self.n_hidden,
            "lr": self.lr
        }
        with open(configs_path, "w") as f:
            json.dump(configs, f)

    def load(self, path, device='cpu'):
        self.model.load_state_dict(torch.load(path, map_location=device))


class LSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, ouput_size):
        super(LSTModel, self).__init__()
        # LSTM cell with a linear layer as a classifier | action predictor. learnable
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, ouput_size)

    def forward(self, x, last_states=None):
        hs, cs = self.cell(x, last_states)
        output = self.linear(hs)
        return output, (hs, cs)
