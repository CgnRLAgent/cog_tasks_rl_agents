import numpy as np
from common.agent import Agent
import os
import json


class Agent_HER(Agent):

    def __init__(self, n_obs, n_act, hierarchy_num, learn_mode='RL', hyperparam={}):

        super(Agent_HER, self).__init__(n_obs, n_act)
        # HER as a classifier | action predictor
        self.model = HERModel(n_obs, n_act, hierarchy_num, learn_mode, hyperparam)
        # internal memory
        self.last_states = None
        self.last_output = None
        # training
        self.is_training = False

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def reset(self): # not sure if HER should update memory gate
        self.last_states = None
        self.model.reset_gate()

    def respond(self, obs):
        # if self.is_training:
        action = self.model.forward(obs=obs, gate='softmax')
        return action

    def learn(self, obs, next_obs, action, reward, done, target_act): 
        r = (reward + 1) / 2  # reward from [-1, 1] to [0, 1]
        self.model.backward(target_act=target_act, reward=r)

    def save(self, dir, name):
        pass

    def load(self, path, device='cpu'):
        pass



class HERModel():
    '''
    create new agent and store
    number of observations
    number of actions
    number of hierarchy layers
    learning mode: 1.RL(reinforcement learning) 2. SL(supervised learning)
    four free hyperparameters for each layer, alpha(learning rate), lambd(Eligibility trace decay),
    beta(working memory update gain), bias(Bias for updating working memory)
    one global hyperparameter gamma(response selection gain)
    '''
    def __init__(self, n_obs, n_act, hierarchy_num, learn_mode='RL', hyperparam={}):

        self.n_obs = n_obs
        self.n_act = n_act
        self.hierarchy_num = hierarchy_num
        self.learn_mode = learn_mode

        self.alpha = hyperparam['alpha']
        self.lambd = hyperparam['lambd']
        self.beta = hyperparam['beta']
        self.bias = hyperparam['bias']
        self.gamma = hyperparam['gamma']

        # initialization
        self.X = self.init_X_weights(n_obs, n_act, hierarchy_num)
        self.W = self.init_W_weights(n_obs, n_act, hierarchy_num)
        self.W_m = {}
        self.d = self.init_d_vector(n_obs, hierarchy_num)

        self.r = {} # the vector of stimuli representation
        self.pred = {}
        self.pred_m = {} # modulated prediction
        self.prob = None # the probability of each action
        self.action = None # the action that HER got in top-down process


    # initialize weights map external stimuli to WM representations
    def init_X_weights(self, n_obs, n_act, hierarchy_num):
        
        X = {}

        # loop all layers
        for index in range(hierarchy_num):
            X[index] = np.zeros((n_obs, n_obs))

        return X


    # initialize weights map vector of task stimulus representations to prediction
    def init_W_weights(self, n_obs, n_act, hierarchy_num):
        
        W = {}

        # each action corresponds to two values(correct and wrong)
        W[0] = np.zeros((2*n_act, n_obs))

        # loop all layers
        # dimension of the row of the last higher weight equal to the dimension of the current weight
        for index in range(1, hierarchy_num):
            row_num = W[index-1].shape[0] * n_obs
            W[index] = np.zeros((row_num, n_obs))

        return W


    # initialize the vector of trace vector
    def init_d_vector(self, n_obs, hierarchy_num):

        d = {}

        for index in range(hierarchy_num):
            d[index] = np.zeros(n_obs)

        return d


    # reset working memory
    def reset_gate(self):
        self.r = {}
        # self.d = self.init_d_vector(self.n_obs, self.hierarchy_num)


    # working memory
    # the mode of gate determines the way to update storing r
    # 1.softmax 2.interpolate 3.max 4.free
    def WM(self, obs, gate='softmax'):

        # the vector of external stimuli
        s = np.zeros(self.n_obs)
        try:
            s[obs] = 1
        except:
            raise ValueError("observation must be the interger from 0 to n_obs-1")

        if self.r == {}: # store the stimuli as r at first time
            for index in range(self.hierarchy_num):
                self.r[index] = s
        else:
            for index in range(self.hierarchy_num):
                r = self.r[index]
                v = np.dot(self.X[index], s) # value of storing
                v_s = v[np.where(s==1)] # value of storing s
                v_r = v[np.where(r==1)] # value of storing r
                if gate == 'free':
                    self.r[index] = s
                else:
                    p_s = np.exp(self.beta[index]*v_s)+self.bias[index]
                    p_r = np.exp(self.beta[index]*v_r)
                    p_storing = p_s / (p_s+p_r)
                    if np.isnan(p_storing): # if probability is nan, set gate as max
                        gate = 'max'
                        # print("The probability of storing new stimuli is nan. Change the mode of gate memory to max")
                    if gate == 'softmax':
                        random_value = np.random.random() # generate a value between 0 and 1
                        if random_value <= p_storing: # check if store new stimuli
                            self.r[index] = s
                    elif gate == 'interpolate':
                        self.r[index] = p_storing*s + (1-p_storing)*r
                    elif gate == 'max':
                        if v_s > v_r:
                            self.r[index] = s
                        # when values are the same, half of the chance to replace r
                        if v_s == v_r and np.random.choice(2, 1):
                            self.r[index] = s
            
        # update d eligibility trace vector
        for index in range(self.hierarchy_num):
            self.d[index] = self.lambd[index]*self.d[index] # gradually decaying
            self.d[index][np.where(s==1)] = 1 # When a stimulus i is presented, the value of di is set to 1


    # softmax function
    def softmax(self, gamma, _vec):

        vec = np.exp(gamma*_vec)
        smax = vec / vec.sum()

        return smax


    # pre-response processing
    def forward(self, obs, gate='softmax'):

        self.WM(obs, gate=gate) # update storing stimuli

        # get prediction
        for index in range(self.hierarchy_num):
            self.pred[index] = np.dot(self.W[index], self.r[index])

        self.pred_m[self.hierarchy_num-1] = self.pred[self.hierarchy_num-1]
        self.W_m[self.hierarchy_num-1] = self.W[self.hierarchy_num-1]

        # top-down process/update pred
        for index in range(self.hierarchy_num-1, 0, -1): # 0 layer is most inferior
            p_matrix = self.pred_m[index].reshape(self.W[index-1].T.shape)
            self.W_m[index-1] = self.W[index-1] + p_matrix.T
            self.pred_m[index-1] = np.dot(self.W_m[index-1], self.r[index-1])
            self.pred_m[index-1] = np.clip(self.pred_m[index-1], 0, 1) # clip [0, 1]

        # compare correct feedback with incorrect feedback to get candidate response
        m = self.pred_m[0]
        u_value = np.zeros((self.n_act))
        for i in range(self.n_act):
            u_value[i] = m[i*2] - m[i*2+1]

        # get the probability by softmax
        self.prob = self.softmax(self.gamma, u_value)

        # response selection
        p_cum = 0
        random_value = np.random.random()
        
        for i, p_u in enumerate(self.prob):
            if random_value <= p_u + p_cum:
                self.action = i
                break
            else:
                p_cum += p_u

        return self.action


    # post-response processing
    # the range of reward should be [0, 1]
    def backward(self, target_act, reward):

        a_filter = np.zeros(2*self.n_act)
        a_filter[(2*self.action):(2*self.action+2)] = 1

        if self.learn_mode == 'RL':
            error = np.multiply(a_filter, reward-self.prob[self.action])
            # don't know how to defore mudulated error in rl!!!!!!!!!!!!!!!
            error_m = np.multiply(a_filter, reward-self.prob[self.action])
        elif self.learn_mode == 'SL':
            pd = self.pred[0]
            pd_m = self.pred_m[0]
            output = np.zeros(pd.shape) # the output of target action
            for i in range(self.n_act):
                if i == target_act:
                    output[2*i] = 1
                else:
                    output[2*i+1] = 1
            error = np.multiply(a_filter, output-pd)
            error_m = np.multiply(a_filter, output-pd_m)

        # bottom-up process
        # update X weight and W weight
        for index in range(self.hierarchy_num):
            # update X weight
            if self.learn_mode == 'RL' and index == 0:
                deltaX = np.outer((reward-self.prob[self.action])*self.r[index], self.d[index])
            else:
                deltaX = np.outer(np.multiply(np.dot(error_m, self.W_m[index]), self.r[index]), self.d[index])
            self.X[index] += deltaX
            # update W weight
            deltaW = self.alpha[index] * np.outer(error_m, self.r[index])
            self.W[index] += deltaW
            # superior layer
            if index < self.hierarchy_num-1:
                # get output of superior layer
                output = np.outer(self.r[index], error)
                output_vector = output.reshape(output.shape[0] * output.shape[1])
                a_filter = np.where(output_vector!=0, 1, 0)
                error = np.multiply(a_filter, output_vector-self.pred[index+1])
                error_m = np.multiply(a_filter, output_vector-self.pred_m[index+1])