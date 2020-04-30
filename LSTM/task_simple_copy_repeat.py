import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test


env = gym.make('Simple_Copy_Repeat-v0', n_char=5, len_range=(2,10), repeat_range=(3,5))

N_tr = 1000
N_tst = 1000
n_hidden = 50
lr = 0.01

agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr)

train(env, agent, N_tr)
test(env, agent, N_tst)