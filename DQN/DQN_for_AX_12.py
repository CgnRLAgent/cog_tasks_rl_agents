#!/usr/bin/env python
# coding: utf-8

# In[72]:


"""
AX_12 TASK:

The AX_12 task consists in the presentation to the subject of six possible stimuli/cues '1' - '2', 'A' - 'B', 'X' - 'Y'.
The tester has 2 possible responses which depend on the temporal order of previous and current stimuli:
he has to answer 'R' when
- the last stored digit is '1' AND the previous stimulus is 'A' AND the current one is 'X',
- the last stored digit is '2' AND the previous stimulus is 'B' AND the current one is 'Y';
in any other case , reply 'L'.

AUTHOR: Zenggo
DATE: 04.2020
"""

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys


class AX_12_ENV(Env):

    DIGITS = ['1', '2']
    CHAR_1 = ['A', 'B', 'C']
    CHAR_2 = ['X', 'Y', 'Z']
    ACTIONS = ['L', 'R']

    def __init__(self, size=10, prob_target=0.3):
        """
        :param size: the length of generated inputs, not including the first digit
        :param prob_target: the probability to generate 'AX' or 'BY'
        """
        # observation (characters)
        self.idx_2_char = self.DIGITS + self.CHAR_1 + self.CHAR_2
        self.char_2_idx = {}
        for i, c in enumerate(self.idx_2_char):
            self.char_2_idx[c] = i
        self.observation_space = Discrete(len(self.idx_2_char))

        # action
        self.action_space = Discrete(len(self.ACTIONS))

        self.size = size // 2
        self.prob_target = prob_target

        # states of an episode
        self.position = None
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = None
        self.input_str = None
        self.target_str = None
        self.output_str = None

        self.np_random = None
        self.seed()
        self.reset()

    @property
    def char_sets(self):
        sets = []
        for c1 in self.CHAR_1:
            for c2 in self.CHAR_2:
                sets.append(c1 + c2)
        return sets

    @property
    def probs(self):
        n_sets = len(self.char_sets)
        prob_other = (1 - self.prob_target) / (n_sets - 2)
        p = np.full(n_sets, prob_other)
        p[self.char_sets.index('AX')] = self.prob_target / 2
        p[self.char_sets.index('BY')] = self.prob_target / 2
        return p

    @property
    def input_length(self):
        return len(self.input_str)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.position = 0
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = 0.0
        self.input_str, self.target_str = self._generate_input_target()
        self.output_str = ''
        obs_char, obs_idx = self._get_observation()
        return obs_idx

    def step(self, action):
        assert self.action_space.contains(action)
        assert 0 <= self.position < self.input_length
        target_act = self.ACTIONS.index(self.target_str[self.position])
        reward = 1.0 if action == target_act else -1.0
        self.last_action = action
        self.last_reward = reward
        self.episode_total_reward += reward
        self.output_str += self.ACTIONS[action]
        self.position += 1
        if self.position < self.input_length:
            done = False
            _, obs = self._get_observation()
        else:
            done = True
            obs = None
        info = {"target_act": target_act}
        return obs, reward, done, info

    def render(self, mode='human'):
        outfile = sys.stdout  #TODO: other mode
        pos = self.position - 1
        o_str = ""
        if pos > -1:
            for i, c in enumerate(self.output_str):
                color = 'green' if self.target_str[i] == c else 'red'
                o_str += colorize(c, color, highlight=True)
        outfile.write("="*20 + "\n")
        outfile.write("Length   : " + str(self.input_length) + "\n")
        outfile.write("Input    : " + self.input_str + "\n")
        outfile.write("Target   : " + self.target_str + "\n")
        outfile.write("Output   : " + o_str + "\n")
        if self.position > 0:
            outfile.write("-" * 20 + "\n")
            outfile.write("Current reward:   %.2f\n" % self.last_reward)
            outfile.write("Cumulative reward:   %.2f\n" % self.episode_total_reward)
        outfile.write("\n")
        return

    def _generate_input_target(self):
        digit = np.random.choice(self.DIGITS)
        input_str = digit
        target_str = 'L'
        for _ in np.arange(self.size):
            s = np.random.choice(self.char_sets, p=self.probs)
            input_str += s
            if digit == '1':
                target_str += 'LR' if s == 'AX' else 'LL'
            else:
                target_str += 'LR' if s == 'BY' else 'LL'
        return input_str, target_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        obs_char = self.input_str[pos]
        obs_idx = self.char_2_idx[obs_char]
        return obs_char, obs_idx


# In[73]:


"""
simple copy task:

Simply copy the input sequence as output. For example:
Input:         ABCDE
Ideal output:  ABCDE

At each time step a character is observed, and the agent should respond a char.
The action(output) is chosen from a char set e.g. {A,B,C,D,E}.

AUTHOR: Zenggo
DATE: 04.2020
"""

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys
import string


class Simple_Copy_ENV(Env):

    ALPHABET = list(string.ascii_uppercase[:26])

    def __init__(self, n_char=5, size=10):
        """
        :param n_char: number of different chars in inputs, e.g. 3 => {A,B,C}
        :param size: the length of input sequence
        """
        self.n_char = n_char
        self.size = size

        # observation (characters)
        self.observation_space = Discrete(n_char)
        # action
        self.action_space = Discrete(n_char)

        # states of an episode
        self.position = None
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = None
        self.input_str = None
        self.target_str = None
        self.output_str = None

        self.np_random = None
        self.seed()
        self.reset()

    @property
    def input_length(self):
        return len(self.input_str)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.position = 0
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = 0.0
        self.input_str, self.target_str = self._generate_input_target()
        self.output_str = ''
        obs_char, obs_idx = self._get_observation()
        return obs_idx

    def step(self, action):
        assert self.action_space.contains(action)
        assert 0 <= self.position < self.input_length
        target_act = self.ALPHABET.index(self.target_str[self.position])
        reward = 1.0 if action == target_act else -1.0
        self.last_action = action
        self.last_reward = reward
        self.episode_total_reward += reward
        self.output_str += self.ALPHABET[action]
        self.position += 1
        if self.position < self.input_length:
            done = False
            _, obs = self._get_observation()
        else:
            done = True
            obs = None
        info = {"target_act": target_act}
        return obs, reward, done, info

    def render(self, mode='human'):
        outfile = sys.stdout  # TODO: other mode
        pos = self.position - 1
        o_str = ""
        if pos > -1:
            for i, c in enumerate(self.output_str):
                color = 'green' if self.target_str[i] == c else 'red'
                o_str += colorize(c, color, highlight=True)
        outfile.write("=" * 20 + "\n")
        outfile.write("Length   : " + str(self.input_length) + "\n")
        outfile.write("Input    : " + self.input_str + "\n")
        outfile.write("Target   : " + self.target_str + "\n")
        outfile.write("Output   : " + o_str + "\n")
        if self.position > 0:
            outfile.write("-" * 20 + "\n")
            outfile.write("Current reward:   %.2f\n" % self.last_reward)
            outfile.write("Cumulative reward:   %.2f\n" % self.episode_total_reward)
        outfile.write("\n")
        return

    def _generate_input_target(self):
        input_str = ""
        for i in range(self.size):
            c = self.np_random.choice(self.ALPHABET[:self.n_char])
            input_str += c
        target_str = input_str
        return input_str, target_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        obs_char = self.input_str[pos]
        obs_idx = self.ALPHABET.index(obs_char)
        return obs_char, obs_idx


# In[74]:


"""
seq_prediction TASK:

Consider two abstract sequences A-B-C-D and X-B-C-Y
In this example remembering that the sequence started with A or X is required 
to make the correct prediction following C. 

AUTHOR: JiqingFeng
DATE: 05.2020
"""


class seq_prediction_ENV(Env):

    STR_in = ['ABC', 'XBC']
    CHAR_in = ['A', 'B', 'C', 'X']
    ACTIONS = ['B', 'C', 'D', 'Y']

    def __init__(self, size=100, p=0.5):
        """
        :param size: the number of inputing stimuli/cues
        :param p: the probability to generate 'ABC' or 'XBC'
        """
        # observation (characters)
        self.idx_2_char = self.CHAR_in
        self.char_2_idx = {}
        for i, c in enumerate(self.idx_2_char):
            self.char_2_idx[c] = i
        self.observation_space = Discrete(len(self.idx_2_char))

        # action
        self.action_space = Discrete(len(self.ACTIONS))

        self.size = size
        self.p = p
        
        # states of an episode
        self.position = None
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = None
        self.input_str = None
        self.target_str = None
        self.output_str = None

        self.np_random = None
        self.seed()
        self.reset()

    @property
    def input_length(self):
        return len(self.input_str)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.position = 0
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = 0.0
        self.input_str, self.target_str = self._generate_input_target(self.size)
        self.output_str = ''
        obs_char, obs_idx = self._get_observation()
        return obs_idx

    def step(self, action):
        assert self.action_space.contains(action)
        assert 0 <= self.position < self.input_length
        target_act = self.ACTIONS.index(self.target_str[self.position])
        reward = 1.0 if action == target_act else -1.0
        self.last_action = action
        self.last_reward = reward
        self.episode_total_reward += reward
        self.output_str += self.ACTIONS[action]
        self.position += 1
        if self.position < self.input_length:
            done = False
            _, obs = self._get_observation()
        else:
            done = True
            obs = None
        info = {"target_act": target_act}
        return obs, reward, done, info

    def render(self, mode='human'):
        outfile = sys.stdout  #TODO: other mode
        pos = self.position - 1
        o_str = ""
        if pos > -1:
            for i, c in enumerate(self.output_str):
                color = 'green' if self.target_str[i] == c else 'red'
                o_str += colorize(c, color, highlight=True)
        outfile.write("="*20 + "\n")
        outfile.write("Length   : " + str(self.input_length) + "\n")
        outfile.write("Input    : " + self.input_str + "\n")
        outfile.write("Target   : " + self.target_str + "\n")
        outfile.write("Output   : " + o_str + "\n")
        if self.position > 0:
            outfile.write("-" * 20 + "\n")
            outfile.write("Current reward:   %.2f\n" % self.last_reward)
            outfile.write("Cumulative reward:   %.2f\n" % self.episode_total_reward)
        outfile.write("\n")
        return

    def _generate_input_target(self, size):
        input_str = ''
        target_str = ''
        for _ in np.arange(int(size/3)):
            s = np.random.choice(self.STR_in, p=[self.p, 1-self.p])
            input_str += s
            if s == 'ABC':
                target_str += 'BCD'
            else:
                target_str += 'BCY'
        remainder = int(size % 3)
        input_str += np.random.choice(self.STR_in, p=[self.p, 1-self.p])[:remainder]
        if remainder == 1:
            target_str += 'B'
        elif remainder == 2:
            target_str += 'BC'
        return input_str, target_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        obs_char = self.input_str[pos]
        obs_idx = self.char_2_idx[obs_char]
        return obs_char, obs_idx


# In[75]:


# The code is borrowed from https://github.com/Leebz/CCE/blob/master/DQN/Copy-v0.py recommended by Jie.
# I made some small changes.

#--------Simple_Copy-----------------------------------
# env = gym.make('Simple_Copy_ENV-v1')
# N_ACTIONS = 5
# N_STATES = 1
#LR = 0.0001
#----------seq_prediction------------------------------
# env = gym.make('seq_prediction_ENV-v1')
# N_ACTIONS = 4
# N_STATES = 1
#LR = 0.0001
#---------AX_12---------------------------------------
# env = gym.make('AX_12_ENV-v1')
# N_ACTIONS = 2
# N_STATES = 1
#LR = 0.001

#---------AX_S_12_ENV---------------------------------------
# env = gym.make('AX_S_12_ENV-v1')
# N_ACTIONS = 2
# N_STATES = 1
#LR = 0.001


#---------AX_CPT_ENV---------------------------------------
# env = gym.make('AX_CPT_ENV-v1')
# N_ACTIONS = 2
# N_STATES = 1
#LR = 0.001
#env.size=20


# ---------AX_12_CPT_ENV---------------------------------------
# env = gym.make('AX_12_CPT_ENV')
# N_ACTIONS = 2
# N_STATES = 1
# LR = 0.001
# env.size=1000


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import balanced_accuracy_score, f1_score
import os
#import sklearn.metrics

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.0001                   # learning rate
DECAY = 0.001
EPSILON_THREAD=0.99
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 200   # target update frequency
MEMORY_CAPACITY = 5000
#env = gym.make('Simple_Copy_ENV-v1')
#env = gym.make('seq_prediction_ENV-v1')
env = AX_12_CPT_ENV()

#env = gym.make('AX_12_CPT_ENV-v1') #env.size=20
#env = gym.make('AX_CPT_ENV-v1') #env.size=20
#env = gym.make('AX_S_12_ENV-v1')#size不设置

env = env.unwrapped
N_ACTIONS = 2  # out_dim
N_STATES = 1  #  in_dim
TRAIN_STEPS = 10000
TEST_STEPS = 1000


S_FOR_DONE=0.0
LAST_ACTION_FOR_ENV_RESET=0
LAST_REWARD_FOR_ENV_RESET=0.0
#env.size=20

#env.render()

# Net(x) = Q(state, action), action = No.0,..,19, x=state
class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc2 = nn.Linear(50, 25)
        self.out = nn.Linear(25, N_ACTIONS)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        l3 = self.out(l2)
        return l3




class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net0(), Net0()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory

        #self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory (state, action, reward, next state)
        self.memory = np.zeros((MEMORY_CAPACITY, 4))     # initialize memory (state, action, reward, next state)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, obs,EPSILON):
        x = torch.unsqueeze(torch.FloatTensor([obs]), 0)  # shape (1,1)
        #print("x",x)

        # input only one sample
        temp_v=np.random.uniform()
        #print("temp_v",temp_v,EPSILON)
        if temp_v< EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # return the argmax index

        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # Q(s_t, a) = r_t + Gamma * argmax_a{Q(s_t+1, a)}
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        #print("q_target",q_target,"b_r",b_r,"GAMMA",GAMMA,"b_s_",b_s_)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def plotLearning(scores):
    N = len(scores)
    x = [i for i in range(N)]
    #plt.ylabel('Score')
    #plt.xlabel('Game(k)')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes/100')
    plt.plot(x, scores)
    plt.show()

def plotAccuracy(accuracys):
    N = len(accuracys)
    x = [i for i in range(N)]
    plt.ylabel('Accuracys')
    plt.xlabel('Episodes/100')
    plt.plot(x, accuracys)
    plt.show()



dqn = DQN()
def train():

    ep_r_avg_trace = []  # record the episodes' rewards

    tr_rewards = []
    tr_accs = []
    tr_f1 = []

    eps_acc = 0  # accuracy of episodes

    ep_r_cum = 0
    EPSILON=0.2
    DECAY=0.001
    for i in range(TRAIN_STEPS):
        temp_times_all = 0
        temp_times_right =0
        acc = 0.0
        s = env.reset()
        env.last_action = LAST_ACTION_FOR_ENV_RESET
        env.last_reward = LAST_REWARD_FOR_ENV_RESET
        done = False
        ep_r = 0  # cummulative reward of current episode


        ep_act_target = []
        ep_act_agent = []
        ep_correct = True

        while not done:
            a = dqn.choose_action(s,EPSILON)
            #target_act = env.ALPHABET.index(env.target_str[env.position])
            #target_act = env.ACTIONS.index(env.target_str[env.position])
            ep_act_agent.append(a)
            s_, r, done, info = env.step(a)
            target_act = info["target_act"]
            ep_act_target.append(target_act)
            if done == True :
                s_=S_FOR_DONE
            dqn.store_transition(s, a, r, s_)
            s = s_
            ep_r += r

            if dqn.memory_counter > MEMORY_CAPACITY :
                dqn.learn()  # train Q
                if EPSILON < EPSILON_THREAD:#贪心e递增
                    EPSILON = EPSILON + DECAY


        ep_r_cum += ep_r
        tr_rewards.append(ep_r)
        ep_acc = balanced_accuracy_score(ep_act_target, ep_act_agent)
        tr_accs.append(ep_acc)
        ep_f1 = f1_score(ep_act_target, ep_act_agent, average='macro')
        tr_f1.append(ep_f1)

    print('\ntraining end.')
    
    return tr_rewards, tr_accs, tr_f1



def test():
    ep_r_avg_trace = []  # record the episodes' rewards
    accuracys =[] # record the episodes' rewards
    accs=[]
    rewards=[]
    f1 = []
    eps_acc = 0  # accuracy of episodes
    test_N=1000
    ep_r_cum = 0
    EPSILON=1.0
    for i in range(test_N):
        temp_times_all = 0
        temp_times_right =0
        acc = 0.0
        s = env.reset()
        env.last_action = LAST_ACTION_FOR_ENV_RESET
        env.last_reward = LAST_REWARD_FOR_ENV_RESET
        done = False
        ep_r = 0  # cummulative reward of current episode


        ep_act_target = []
        ep_act_agent = []
        ep_correct = True



        while not done:
            a = dqn.choose_action(s,EPSILON)
            #target_act = env.ALPHABET.index(env.target_str[env.position])
            #target_act = env.ACTIONS.index(env.target_str[env.position])
            ep_act_agent.append(a)
            s_, r, done, info = env.step(a)
            target_act = info["target_act"]
            ep_act_target.append(target_act)
            if done == True :
                s_=S_FOR_DONE
            dqn.store_transition(s, a, r, s_)
            s = s_
            ep_r += r
            if target_act != a:
                ep_correct = False

        ep_r_cum += ep_r
        rewards.append(ep_r)
        ep_acc = balanced_accuracy_score(ep_act_target, ep_act_agent)
        accs.append(ep_acc)
        ep_f1 = f1_score(ep_act_target, ep_act_agent, average='macro')
        f1.append(ep_f1)
        if ep_correct:
            eps_acc += 1


    avg_reward = np.mean(rewards)
    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1)


    eps_acc /= test_N

    print('\ntest end.')
    print('test episode accuracy: %.2f, avg reward: %.2f, avg accuracy: %.4f, avg f1: %.4f' % (eps_acc, avg_reward, avg_acc, avg_f1))
    
    return avg_reward, avg_acc, avg_f1


if __name__ == '__main__':
    dqn = DQN()
    results = train()
    avg_reward, avg_acc, avg_f1 = test()


# In[76]:


def save_train_res(path, results):
    """
    :param results: return of train()
    """
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    np.save(path, np.array(results), allow_pickle=True)


# In[77]:


p = "/Users/neil/Documents/Results_DQN/results_of_AX_12"

save_train_res(p, results)


# In[78]:


def load_train_res(path):
    r = np.load(path, allow_pickle=True)
    return r[0], r[1], r[2]


# In[79]:


def train_results_plots(dir, figname, names, numbers, smooth=51):
    """
    plots training results with iterations: rewards, accuracies, f1-score (every n iterations)
    :param dir: save the figures to
    :param figname
    :param names: [str, ...] the names of the agents to be compared
    :param numbers: [(rewards, accs), ...] the training results
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    figname = os.path.join(dir, figname)

    def _smooth(p):
        r = (smooth-1) // 2
        sp = p[:]
        size = len(p)
        for i in range(size):
            begin = np.max([0, i-r])
            end = np.min([size-1, i+r]) + 1
            sp[i] = np.mean(p[begin:end])
        return sp

    # plot rewards
    plt.figure(figsize=(14, 8))
    rewards = [x[0] for x in numbers]
    assert len(rewards) == len(names)
    plt.title('Rewards')
    for r in rewards:
        plt.plot(_smooth(r))
    plt.legend(names, loc='lower right')
    plt.xlabel('iterations')
    plt.savefig(figname + '_rewards.jpg')
    # plot accuracies
    plt.figure(figsize=(14, 8))
    accs = [x[1] for x in numbers]
    assert len(accs) == len(names)
    plt.title('Accuracy')
    for a in accs:
        plt.plot(_smooth(a))
    plt.legend(names, loc='lower right')
    plt.xlabel('iterations')
    plt.savefig(figname + '_accs.jpg')
    # plot f1
    plt.figure(figsize=(14, 8))
    f1 = [x[2] for x in numbers]
    assert len(f1) == len(names)
    plt.title('F1-score')
    for f in f1:
        plt.plot(_smooth(f))
    plt.legend(names, loc='lower right')
    plt.xlabel('iterations')
    plt.savefig(figname + '_f1.jpg')


# In[80]:


res = load_train_res('/Users/neil/Documents/Results_DQN/results_of_AX_12.npy')
train_results_plots(dir = '/Users/neil/Documents/Results_DQN/' , figname='AX_12_ENV', names=['DQN'],                     numbers=[res])


# In[ ]:




