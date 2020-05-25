import gym
import gym_cog_ml_tasks
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from HER.HER_model import Agent_HER
from common.utils import train, test, save_train_res, load_train_res, train_results_plots
import numpy as np


env = gym.make('Simple_Copy-v0', size=10)

N_tr = 50
N_tst = 1000
hierarchy_num = 1 # size*repeat
learn_mode = 'SL'
hyperparam = {
    'alpha':np.ones(hierarchy_num) * 0.075,
    'lambd':np.linspace(0.1, 0.99, hierarchy_num),
    'beta':np.ones(hierarchy_num) * 15,
    'bias':np.array([1/(10**i) for i in range(hierarchy_num)]),
    'gamma':15
}
agent = Agent_HER(env.observation_space.n, env.action_space.n, hierarchy_num, learn_mode, hyperparam)

res = train(env, agent, N_tr, seed=123)
# save the training records, including every episode's reward, action accuracy and f1 over iteration
save_train_res('./save/simple_copy/HER_10', res)
# test(env, agent, N_tst, seed=123)