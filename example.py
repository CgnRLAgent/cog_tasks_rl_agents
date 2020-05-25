import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from HER.HER_model import Agent_HER
from common.utils import train, test, save_train_res, load_train_res, train_results_plots
import torch
import numpy as np

torch.manual_seed(123)

env = gym.make('12_AX-v0', size=10, prob_target=0.5)

N_tr = 2000
N_tst = 1000

n_hidden = 10
n_layers = 1
lr = 0.01
lstm = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr, n_layers)

hierarchy_num = 3
learn_mode = 'SL'
hyperparam = {
    'alpha': np.ones(hierarchy_num) * 0.075,
    'lambd': np.array([0.1, 0.5, 0.99]),
    'beta': np.ones(hierarchy_num) * 15,
    'bias': np.array([1/(10**i) for i in range(hierarchy_num)]),
    'gamma': 15
}
her = Agent_HER(env.observation_space.n, env.action_space.n, hierarchy_num, learn_mode, hyperparam)



res_lstm = train(env, lstm, N_tr, seed=123)
res_her = train(env, her, N_tr, seed=123)
# save the training records, including every episode's reward, action accuracy and f1 over iteration
save_train_res('./save/LSTM', res_lstm)
save_train_res('./save/HER', res_her)

test(env, lstm, N_tst, seed=123)
test(env, her, N_tst, seed=123)

res_lstm = load_train_res('./save/LSTM.npy')
res_her = load_train_res('./save/HER.npy')
train_results_plots(dir='./save/', figname='test', names=['LSTM', 'HER'], \
                    numbers=[res_lstm, res_her])