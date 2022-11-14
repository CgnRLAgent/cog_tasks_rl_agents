import torch
import torch.nn as nn
from torch.optim import Adam
from common.agent import Agent
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent_LSTM(Agent):
    def __init__(self, n_obs, n_act, n_hidden, lr=0.01, n_layers=1):
        super(Agent_LSTM, self).__init__(n_obs, n_act)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        # LSTM as a classifier | action predictor
        self.model = LSTModel(n_obs, n_hidden, n_act, n_layers)
        self.model.to(device)
        print("Agent works on %s" % device)
        # reset internal memory
        self.reset()
        # training
        self.is_training = False
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        
        # needed to get traceback on any inconsistencies in the graph during backprop
        torch.autograd.set_detect_anomaly(True)

    def train(self):
        self.model.train()
        self.is_training = True

    def eval(self):
        self.model.eval()
        self.is_training = False

    def reset(self):
        self.last_states_list = [None]
        self.outputs = []

    def respond(self, obs):
        x = torch.tensor([obs]).to(device)
        if self.is_training:
            output, last_states = self.model(x, self.last_states_list[-1])
        else:
            with torch.no_grad():
                output, last_states = self.model(x, self.last_states_list[-1])
        self.outputs.append(output.to('cpu'))
        self.last_states_list.append(last_states)
        action = self.outputs[-1].argmax().item()
        return action

    def learn(self, obs, next_obs, action, reward, done, target_act):
        loss = self.criterion(self.outputs[-1], torch.tensor([target_act]))
        loss.backward(retain_graph=True)
        ## optimizer.step() changes weights in-place,
        ##  and then a backward on the next time step gives error since pytorch version > 1.4
        ##  see: https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/45
        ## so only do optimizer.step() if episode is done.
        ## gradient will accumulate with backward in every time step.
        if done:
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

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))


class LSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        # LSTM layers with a linear layer as a classifier(action predictor)
        self.layers = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for _ in range(n_layers)])
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, last_states=None):
        N = len(self.layers)
        if last_states is not None:
            assert len(last_states) == N
        else:
            last_states = [None] * N

        states = []
        #states = torch.empty((N,2,1,self.hidden_size), device=device)
        hin = self.embedding(x)

        for n in range(N):
            lstm = self.layers[n]
            ht, ct = lstm(hin, last_states[n])
            states.append((ht, ct))
            hin = ht

        output = self.linear(states[-1][0])
        #print('e=',output)
        return output, states
