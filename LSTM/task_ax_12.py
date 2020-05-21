import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test
import torch

torch.manual_seed(123)

env = gym.make('AX_12-v0', size=10, prob_target=0.5)

N_tr = 200
N_tst = 100
n_hidden = 20
n_layers = 1
lr = 0.01

agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr, n_layers)

train(env, agent, N_tr, seed=123)
test(env, agent, N_tst)