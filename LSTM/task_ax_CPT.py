import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test, save_train_res, load_train_res, train_results_plots


env = gym.make('AX_CPT-v0', size=50)

N_tr = 500
N_tst = 100
n_hidden = 50
lr = 0.01
agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr)

res = train(env, agent, N_tr, print_progress=True, seed=123)
save_train_res('./save/ax_cpt/LSTM_50_res', res)

train_results_plots(dir='./save/ax_cpt', figname='LSTM_50', names=['LSTM_50'], numbers=[res])