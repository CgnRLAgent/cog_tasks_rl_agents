import gym
import gym_cog_ml_tasks
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from MonteCarlo.MonteCarlo_model import Agent_MC
from common.utils import train, test, save_train_res, train_results_plots
import torch
import numpy as np

res = np.load('./save/simple_copy/MC_3.npy')

train_results_plots(dir='./save/simple_copy', figname='MC_3', names=['MC_3'], numbers=[res])
