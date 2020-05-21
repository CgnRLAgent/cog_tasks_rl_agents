import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test, save_train_res, load_train_res, train_results_plots
import torch
torch.manual_seed(123)

env = gym.make('seq_prediction-v0', size=30, p=0.5)

N_tr = 100
N_tst = 100
n_hidden = 20
n_layers = 1
lr = 0.01
agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr, n_layers)

res_1 = train(env, agent, N_tr, print_progress=True, seed=123)
save_train_res('./save/seq_pred/LSTM_l1_res', res_1)
# agent.save(dir='./save/seq_pred', name='LSTM_10')
# agent.load('./save/seq_pred/LSTM_10')
test(env, agent, N_tst, print_progress=True, seed=123)
# a = load_train_res('./save/seq_pred/LSTM_l1_res.npy')


n_layers = 2
agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr, n_layers)

res_2 = train(env, agent, N_tr, print_progress=True, seed=123)
save_train_res('./save/seq_pred/LSTM_l2_res', res_2)
# agent.save(dir='./save/seq_pred', name='LSTM_30')
# test(env, agent, N_tst, print_progress=True, seed=123)

train_results_plots(dir='./save/seq_pred', figname='LSTM', names=['LSTM_l1', 'LSTM_l2'], numbers=[res_1, res_2])