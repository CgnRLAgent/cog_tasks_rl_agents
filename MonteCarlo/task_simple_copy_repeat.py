import gym
import gym_cog_ml_tasks
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from MonteCarlo.MonteCarlo_model import Agent_MC
from common.utils import train, test, save_train_res, train_results_plots
import torch

seed = 123

env = gym.make('Simple_Copy_Repeat-v0', n_char=5, size=10, repeat=3)

N_tr = 10000
N_tst = 1000



agent = Agent_MC(env.observation_space.n, env.action_space.n)
res = train(env, agent, N_tr, seed=123, print_progress=True)
test(env, agent, N_tst, seed=123, print_progress=True)
save_train_res('./agents/cog_tasks_rl_agents/MonteCarlo/save/copy_repeat/MC_10_3', res)
train_results_plots(dir='./agents/cog_tasks_rl_agents/MonteCarlo/save/copy_repeat/', figname='MC_10_3', names=['MC_10_3'], numbers=[res])