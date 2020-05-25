import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test, save_train_res
import torch

torch.manual_seed(123)

env = gym.make('AX_CPT-v0', size=100, prob_target=0.5)

N_tr = 20
N_tst = 1000

n_hidden = 10
n_layers = 1
lr = 0.01
agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr, n_layers)

res = train(env, agent, N_tr, seed=123)
# save the training records, including every episode's reward, action accuracy and f1 over iteration
save_train_res('./save/ax_cpt/LSTM', res)
test(env, agent, N_tst, seed=123)