import gym
import gym_cog_ml_tasks
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.utils import train, test, save_train_res, train_results_plots
from MonteCarlo_model import Agent_MC
import torch

torch.manual_seed(123)
env = gym.make('12_AX_S-v0', size=10,prob_target=0.5)

seed = 123
N_tr = 50000
N_tst = 1000


agent = Agent_MC(env.observation_space.n, env.action_space.n)

res = train(env, agent, N_tr, custom_reward=lambda r: r*10, seed = seed)
test(env, agent, N_tst,seed=seed)
save_train_res('./agents/cog_tasks_rl_agents/MonteCarlo/save/12_ax_s/MC_10_0.5', res)
train_results_plots(dir='./agents/cog_tasks_rl_agents/MonteCarlo/save/12_ax_s/', figname='MC_10_0.5', names=['MC_10_0.5'], numbers=[res])
