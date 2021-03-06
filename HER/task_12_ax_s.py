import gym
import gym_cog_ml_tasks
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from HER.HER_model import Agent_HER
from common.utils import train, test, save_train_res, load_train_res, train_results_plots
import numpy as np


env = gym.make('12_AX_S-v0', size=10, prob_target=0.5)

N_tr = 220
N_tst = 1000
hierarchy_num = 3
learn_mode = 'SL'
hyperparam = {
    'alpha':np.ones(hierarchy_num) * 0.075,
    'lambd':np.array([0.1, 0.5, 0.99]),
    'beta':np.ones(hierarchy_num) * 15, 
    'bias':np.array([1/(10**i) for i in range(hierarchy_num)]),
    'gamma':15
}
agent = Agent_HER(env.observation_space.n, env.action_space.n, hierarchy_num, learn_mode, hyperparam)

res = train(env, agent, N_tr, seed=123)
# save the training records, including every episode's reward, action accuracy and f1 over iteration
save_train_res('./save/12_ax_s/HER', res)
# test(env, agent, N_tst, seed=123)