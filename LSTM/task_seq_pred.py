import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
from common.utils import train, test


env = gym.make('seq_prediction-v0', size=10, p=0.5)

N_tr = 300
N_tst = 100
n_hidden = 10
lr = 0.01

agent = Agent_LSTM(env.observation_space.n, env.action_space.n, n_hidden, lr)

train(env, agent, N_tr)
# agent.load('./save/seq_pred_params')
test(env, agent, N_tst)

# agent.save(dir='./save', name='seq_pred')