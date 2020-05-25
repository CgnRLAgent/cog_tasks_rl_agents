import gym
import gym_cog_ml_tasks
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from AuGMEnT_model import Agent_AuGMEnT
from common.utils import train, test, save_train_res, load_train_res, train_results_plots
import numpy as np

env = gym.make('seq_prediction-v0', size=50, p=0.5)

N_tr = 10000
N_tst = 1000

S = 8        			# dimension of the input = number of possible stimuli
R = 3			     	# dimension of the regular units
M = 4 			     	# dimension of the memory units
A = 2			     	# dimension of the activity units = number of possible responses
	
# value parameters were taken from the 
lamb = 0.15    			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0.025		# percentage of softmax modality for activity selection
g = 1

leak = 1.0			# additional parameter: leaking decay of the integrative memory

# reward settings
rew = 'SRL'
prop = 'std'

policy_train = 'eps_greedy'
policy_test = 'greedy'
stoc_train = 'soft'
stoc_test = 'unif'
t_weighted_train = True
t_weighted_test = False
e_weighted = False
first_flag = False
reset_tags_seq = True

agent = Agent_AuGMEnT(env.observation_space.n, R, M, env.action_space.n, alpha, beta, discount, eps, g, leak, rew, prop, policy_train, policy_test, stoc_train, stoc_test, t_weighted_train, t_weighted_test, e_weighted, first_flag, reset_tags_seq)

res=train(env,agent,N_tr,seed=123)
# save the training records, including every episode's reward, action accuracy and f1 over iteration
save_train_res('save/seq_pred/AuGMEnT_size_50',res)
test(env,agent,N_tst,seed=123)