import gym
import gym_cog_ml_tasks
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from MonteCarlo.MonteCarlo_model import Agent_MC
from common.utils import train, test, save_train_res,train_results_plots


env = gym.make('seq_prediction-v0', size=50, p=0.5)

N_tr = 80000
N_tst = 1000

agent = Agent_MC(env.observation_space.n, env.action_space.n)

res = train(env, agent, N_tr, print_progress=True, seed=123)
test(env, agent, N_tst, print_progress=True, seed=123)
save_train_res('./agents/cog_tasks_rl_agents/MonteCarlo/save/seq_pred/MC_50_.5', res)
train_results_plots(dir='./agents/cog_tasks_rl_agents/MonteCarlo/save/seq_pred', figname='MC_50_.5', names=['MC_50_.5'], numbers=[res])


