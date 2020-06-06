import gym
import gym_cog_ml_tasks
from DRQN.DRQN_agent import Agent_DRQN
from common.utils import train, test, save_train_res
import torch

torch.manual_seed(123)

env = gym.make('12_AX_CPT-v0', size=100, prob_target=0.5, prob_12=0.1)

N_tr = 10000
N_tst = 1000

max_mem_size = 300
lr = 1e-3
epsilon = 0.999
gamma = 0.9
drqn_params = {
    "lstm_hidden_size": 50,
    "n_lstm_layers": 1,
    "linear_hidden_size": 50,
    "n_linear_layers": 1
}

agent = Agent_DRQN(env.observation_space.n, env.action_space.n, max_mem_size, lr, epsilon, gamma, drqn_params)

res = train(env, agent, N_tr, seed=123)
save_train_res('./save/12_ax_cpt/DRQN', res)
test(env, agent, N_tst, seed=123)