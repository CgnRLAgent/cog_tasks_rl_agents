import gym
import gym_cog_ml_tasks
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from MonteCarlo.MonteCarlo_model import Agent_MC
from common.utils import train, test, save_train_res, train_results_plots
import torch


env = gym.make('12_AX_CPT-v0', size=100, prob_target=0.5, prob_12 =0.1)

N_tr = 50000
N_tst = 1000

agent = Agent_MC(env.observation_space.n, env.action_space.n)

res = train(env, agent, N_tr, print_progress=True, seed=123)
test(env, agent, N_tst, seed=123)
save_train_res('./agents/cog_tasks_rl_agents/MonteCarlo/save/12_ax_CPT/MC_10_0.5_0.1', res)
train_results_plots(dir='./agents/cog_tasks_rl_agents/MonteCarlo/save/12_ax_CPT/', figname='MC_100_0.5_0.1', names=['MC_100_0.5_0.1'], numbers=[res])