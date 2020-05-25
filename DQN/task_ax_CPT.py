import gym
import gym_cog_ml_tasks
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from DQN.DQN_model import Agent_DQN
from common.utils import train, test, save_train_res, load_train_res, train_results_plots
import numpy as np
env = gym.make('AX_CPT-v0', size=100)

N_tr = 2000
N_tst = 1000

BATCH_SIZE = 32
LR = 0.001                  # learning rate
DECAY = 0.001
EPSILON = 0.2               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 200   # target update frequency
MEMORY_CAPACITY = 5000
N_ACTIONS = env.action_space.n
N_STATES = 1
S_FOR_DONE = 0.0

agent = Agent_DQN(env.observation_space.n, env.action_space.n, MEMORY_CAPACITY, N_STATES, LR, EPSILON, N_ACTIONS, TARGET_REPLACE_ITER, BATCH_SIZE, GAMMA, DECAY, S_FOR_DONE)

# train(env, agent, N_tr, seed=123)
# test(env, agent, N_tst, seed=123)

res=train(env,agent,N_tr,seed=123)
# save the training records, including every episode's reward, action accuracy and f1 over iteration
save_train_res('./save/ax_cpt/DQN',res)
test(env,agent,N_tst,seed=123)