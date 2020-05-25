import gym
import gym_cog_ml_tasks
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from MonteCarlo.MonteCarlo_model import Agent_MC
from common.utils import train, test, save_train_res, train_results_plots
import torch


env = gym.make('AX_CPT-v0', size=100, prob_target=0.5)

N_tr = 50000
N_tst = 1000

agent = Agent_MC(env.observation_space.n, env.action_space.n)

res = train(env, agent, N_tr, print_progress=True, seed=123)
test(env, agent, N_tst, print_progress=True, seed=123)
save_train_res('./agents/cog_tasks_rl_agents/MonteCarlo/save/ax_CPT/MC_100_0.5', res)
train_results_plots(dir='./agents/cog_tasks_rl_agents/MonteCarlo/save/ax_CPT/', figname='MC_100_0.5', names=['MC_100_0.5'], numbers=[res])
