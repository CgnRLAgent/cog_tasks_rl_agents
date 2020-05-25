import numpy as np
from common.agent import Agent
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent_DQN(Agent):

    def __init__(self, n_obs, n_act, MEMORY_CAPACITY, N_STATES, LR, EPSILON, N_ACTIONS, TARGET_REPLACE_ITER, BATCH_SIZE, GAMMA, DECAY, S_FOR_DONE):

        super(Agent_DQN, self).__init__(n_obs, n_act)
        # DQN as a classifier | action predictor
        self.model = DQN(MEMORY_CAPACITY, N_STATES, LR, EPSILON, N_ACTIONS, TARGET_REPLACE_ITER, BATCH_SIZE, GAMMA)
        # internal memory
        self.last_states = None
        self.last_output = None
        # training
        self.is_training = False
        # decay exploiting ramdom action
        self.DECAY = DECAY
        # store (obs, reward, action, next_obs)
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        # 
        self.S_FOR_DONE = S_FOR_DONE

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def reset(self):
        self.last_states = None
        self.last_output = None

    def respond(self, obs):
        if self.is_training == False:
            self.model.EPSILON = 1
        action = self.model.choose_action(obs)
        return action

    def learn(self, obs, next_obs, action, reward, done, target_act):
        if done == True :
            next_obs = self.S_FOR_DONE
        self.model.store_transition(obs, action, reward, next_obs)
        if self.model.memory_counter > self.MEMORY_CAPACITY:
            self.model.learn()
            if self.model.EPSILON < 0.99:
                self.model.EPSILON += self.DECAY

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


class DQN(object):
    def __init__(self, MEMORY_CAPACITY, N_STATES, LR, EPSILON, N_ACTIONS, TARGET_REPLACE_ITER, BATCH_SIZE, GAMMA):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory (state, action, reward, next state)
        # self.memory = np.zeros((MEMORY_CAPACITY, 4))     # initialize memory (state, action, reward, next state)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        self.EPSILON = EPSILON
        self.N_ACTIONS = N_ACTIONS
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.BATCH_SIZE = BATCH_SIZE
        self.N_STATES = N_STATES
        self.GAMMA = GAMMA

    def choose_action(self, obs):
        x = torch.unsqueeze(torch.FloatTensor([obs]), 0)  # shape (1,1)
        # input only one sample
        if np.random.uniform() < self.EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        n_state = self.N_STATES
        b_s = torch.FloatTensor(b_memory[:, :n_state]) # sometimes may need several elements to represent states
        b_a = torch.LongTensor(b_memory[:, n_state:n_state+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, n_state+1:n_state+2])
        b_s_ = torch.FloatTensor(b_memory[:, -n_state:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) Q(s_t, a_t) 
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # Q(s_t, a) = r_t + Gamma * argmax_a{Q(s_t+1, a)}
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# speed of convergence is high than Net
class Net0(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc2 = nn.Linear(50, 25)
        self.out = nn.Linear(25, N_ACTIONS)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        l3 = self.out(l2)
        return l3

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization weight (mean and variance)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value