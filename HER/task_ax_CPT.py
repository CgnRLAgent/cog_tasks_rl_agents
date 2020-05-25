import gym
import gym_cog_ml_tasks
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from HER.HER_model import Agent_HER
from common.utils import train, test, save_train_res, load_train_res, train_results_plots
import numpy as np


env = gym.make('AX_CPT-v0', size=100, prob_target=0.5)

N_tr = 30
N_tst = 1000
hierarchy_num = 2
learn_mode = 'SL'
hyperparam = {
    'alpha':np.ones(hierarchy_num) * 0.075,
    'lambd':np.array([0.1, 0.5]),
    'beta':np.ones(hierarchy_num) * 15,
    'bias':np.array([1/(10**i) for i in range(hierarchy_num)]),
    'gamma':15
}
agent = Agent_HER(env.observation_space.n, env.action_space.n, hierarchy_num, learn_mode, hyperparam)

res = train(env, agent, N_tr, seed=123)
# save the training records, including every episode's reward, action accuracy and f1 over iteration
save_train_res('./save/ax_cpt/HER', res)
# test(env, agent, N_tst, seed=123)