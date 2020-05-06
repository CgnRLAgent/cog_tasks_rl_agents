import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test, save_train_res, load_train_res, train_results_plots


env = gym.make('seq_prediction-v0', size=10, p=0.5)

N_tr = 300
N_tst = 100
n_hidden = 10
lr = 0.01
agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr)

res_10 = train(env, agent, N_tr, print_progress=True, seed=123)
save_train_res('./save/seq_pred/LSTM_10_res', res_10)
# agent.save(dir='./save/seq_pred', name='LSTM_10')
# agent.load('./save/seq_pred/LSTM_10')
# test(env, agent, N_tst, print_progress=True, seed=123)
# a = load_train_res('./save/seq_pred/LSTM_10_res.npy')


N_tr = 300
N_tst = 100
n_hidden = 30
lr = 0.01
agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr)

res_30 = train(env, agent, N_tr, print_progress=True, seed=123)
save_train_res('./save/seq_pred/LSTM_30_res', res_30)
# agent.save(dir='./save/seq_pred', name='LSTM_30')
# test(env, agent, N_tst, print_progress=True, seed=123)

train_results_plots(dir='./save/seq_pred', figname='LSTM', names=['LSTM_10', 'LSTM_30'], numbers=[res_10, res_30])