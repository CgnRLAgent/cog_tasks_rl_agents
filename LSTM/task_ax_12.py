import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test


env = gym.make('AX_12-v0', size=10, prob_target=0.5)

N_tr = 1000
N_tst = 100
n_hidden = 20
lr = 0.01

agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr)

train(env, agent, N_tr)
test(env, agent, N_tst)