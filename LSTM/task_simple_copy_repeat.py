import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test
import torch
torch.manual_seed(123)

env = gym.make('Simple_Copy_Repeat-v0', n_char=5, size=3, repeat=2)

N_tr = 3000
N_tst = 100

n_hidden = 100
n_layers = 2
lr = 0.01

agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr, n_layers)
train(env, agent, N_tr, seed=123, print_progress=True)
test(env, agent, N_tst, seed=123, print_progress=True)