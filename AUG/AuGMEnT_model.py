import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.agent import Agent
import json
import math


class Agent_AuGMEnT(Agent):

    def __init__(self, n_obs, R, M, n_act, alpha, beta, discount, eps, g, leak, rew, prop, policy_train='eps_greedy', policy_test='greedy', stoc_train='unif', stoc_test='unif', t_weighted_train=False, t_weighted_test=False, e_weighted=False, first_flag=None, reset_tags_seq=False):

        super(Agent_AuGMEnT, self).__init__(n_obs, n_act)
        # HER as a classifier | action predictor
        self.model = AuGMEnT(n_obs, R, M, n_act, alpha, beta, discount, eps, g, leak, rew, prop)
        # internal memory
        self.last_states = None
        self.last_output = None
        # training
        self.is_training = False

        self.episode_num = 0
        self.start_test = False

        self.policy_train = policy_train
        self.policy_test = policy_test
        self.stoc_train = stoc_train
        self.stoc_test = stoc_test
        self.t_weighted_train = t_weighted_train
        self.t_weighted_test = t_weighted_test
        self.e_weighted = e_weighted
        self.model.first = first_flag
        self.reset_tags_seq = reset_tags_seq

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def reset(self):
        self.last_states = None
        self.model.reset_memory()
        if self.is_training or self.reset_tags_seq:
            self.model.reset_tags()
        # if self.is_training and self.reset_tags_seq:
        #    self.model.r = 0
        if self.is_training and self.model.first != None:
            self.model.first = True

    def respond(self, obs):
        if self.is_training:
            action = self.model.get_action(obs, self.episode_num, self.policy_train, self.stoc_train, self.t_weighted_train, self.e_weighted)
        else:
            action = self.model.get_action(obs, self.episode_num, self.policy_test, self.stoc_test, self.t_weighted_test, self.e_weighted)
            if self.start_test == False:
                self.model.initialize_s_old()
                self.start_test = True
        return action

    def learn(self, obs, next_obs, action, reward, done, target_act):
        self.episode_num += 1
        self.model.learn(obs,target_act)

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        params_path = os.path.join(dir, name)
        torch.save(self.model.state_dict(), params_path)
        configs_path = os.path.join(dir, name+'_configs.json')
        configs = {
            "n_obs": self.n_obs,
            "n_act": self.n_act,
            "n_hidden": self.n_hidden,
            "lr": self.lr
        }
        with open(configs_path, "w") as f:
            json.dump(configs, f)

    def load(self, path, device='cpu'):
        self.model.load_state_dict(torch.load(path, map_location=device))



class AuGMEnT():

    ## Inputs
    # ------------
    # S: int, dimension of the input stimulus for both the instantaneous and transient units
    # R: int, number of neurons for the regular units
    # M: int, number of units in the memory layer
    # A: int, number of activity units

    # alpha: scalar, decay constant of synaptic tags (< 1)
    # beta: scalar, gain parameter for update rules
    # discount: scalar, discount dactor
    # epsilon: scalar, response exploration parameter
    # gain: scalar, concentration parameter for response selection
    # leak: scalar or list, leak of the memory dynamics (if a list, the memory units are divided in groups of same size with different leak rates)
        
    # rew_rule: string, defining the rewarding system for correct and wrong predictions ('RL','PL','SRL','BRL')
    # dic_stim: dictionary, with associations stimulus-label
    # dic_resp: dictionary, with associations response-label
    # prop: string, propagation system ('std','BP','RBP','SRBP','MRBP')

    def __init__(self,S,R,M,A,alpha,beta,discount,eps,gain,leak,rew_rule='RL',prop='std'):
 
        self.S = S
        self.R = R
        self.M = M
        self.A = A

        self.alpha = alpha
        self.beta = beta
        self.discount = discount
        self.epsilon = eps
        self.gain = gain

        self.memory_leak = leak

        if isinstance(self.memory_leak, list):
            if np.remainder(self.M,len(self.memory_leak))!=0:
                raise ValueError('Size of the leak vector is not compliant to the number of memory units.')
        
        self.prop = prop

        self.define_reward_rule(rew_rule)

        self.initialize_weights_and_tags()

        self.r = None
        self.q_old = None

        self.initialize_s_old()
        self.s_inst = None
        self.s_trans = None
                    
        self.y_r = None
        self.y_m = None
        self.Q = None

        self.resp_ind = None

    def initialize_s_old(self):
        zero = np.zeros((1,self.S))
        self.s_old = zero

    def initialize_weights_and_tags(self):

        rang = 1

        if self.prop=='std' or self.prop=='RBP' or self.prop=='SRBP' or self.prop=='MRBP':
            
            self.V_r = rang*np.random.random((self.S,self.R)) - rang/2
            self.W_r = rang*np.random.random((self.R,self.A)) - rang/2

            self.V_m = rang*np.random.random((2*self.S,self.M)) - rang/2
            self.W_m = rang*np.random.random((self.M,self.A)) - rang/2

            self.W_r_back = rang*np.random.random((self.A,self.R)) - rang/2
            self.W_m_back = rang*np.random.random((self.A,self.M)) - rang/2

        elif self.prop=='BP':

            self.V_r = rang*np.random.random((self.S,self.R)) - rang/2
            self.W_r = rang*np.random.random((self.R,self.A)) - rang/2

            self.V_m = rang*np.random.random((2*self.S,self.M)) - rang/2
            self.W_m = rang*np.random.random((self.M,self.A)) - rang/2
            
            self.W_r_back = np.transpose(self.W_r)
            self.W_m_back = np.transpose(self.W_m)

        self.reset_memory()
        self.reset_tags()

    def define_reward_rule(self,rew_rule, seq_pred=False):

        if rew_rule =='RL':
            self.rew_pos = 1
            self.rew_neg = 0
        elif rew_rule =='PL':
            self.rew_pos = 0
            self.rew_neg = -1
        elif rew_rule =='SRL':
            self.rew_pos = 1
            self.rew_neg = -1
        elif rew_rule =='BRL':
            self.rew_pos = 0.1
            self.rew_neg = -1


    def positive_reward_seq_pred(self,final,d=3):
        
        if final==True:
            return 1.5
        else:
            #return (d-1.5)/(d-1)
            return 0.75

    def compute_response(self, Qvec, policy='eps_greedy',stoc='unif',t_weighted=False,e_weighted=False,it=None):

        resp_ind = None
        P_vec = None
        
        if policy=='eps_greedy':
        
            if e_weighted==True and it is not None:
                eps = self.epsilon*(1-(2/np.pi)*np.arctan(it/2000))
            else:
                eps = self.epsilon
            # greedy
            if np.random.random()<=(1-eps):
                if len(np.where(np.squeeze(Qvec)==np.max(Qvec)) )==1:
                    resp_ind = np.argmax(Qvec)
                else:
                    resp_ind = np.random.choice(np.arange(len(Qvec)),1).item()
            else:
                if stoc=='soft':
                    if t_weighted==True and it is not None:
                        g = 1 + (10/np.pi)*np.arctan(it/2000)
                    else:
                        g = 1
                    tot = np.clip(a=g*Qvec,a_min=-500,a_max=500)
                    P_vec = np.exp(tot)
                    if (np.isnan(P_vec)).any()==True:
                        resp_ind = np.argmax(Qvec)
                    else:
                        P_vec = P_vec/np.sum(P_vec)
                        resp_ind = np.random.choice(np.arange(len(np.squeeze(Qvec))),1,p=np.squeeze(P_vec)).item()
                elif stoc=='unif':
                    resp_ind = np.random.choice(np.arange(len(np.squeeze(Qvec))),1).item()
        
        elif policy=='greedy':
            #print('GREEDY')
            if len(np.where(np.squeeze(Qvec)==np.max(Qvec)) )==1:
                resp_ind = np.argmax(Qvec)
            else:
                resp_ind = np.random.choice(np.arange(len(Qvec)),1).item()
                
        elif policy=='softmax':
            #print('SOFTMAX')
            if t_weighted==True and it is not None:
                g = 1 + (10/np.pi)*np.arctan(it/2000)
            else:
                g = 1
            tot = np.clip(a=g*Qvec,a_min=-500,a_max=500)
            P_vec = np.exp(tot)
            if (np.isnan(P_vec)).any()==True:
                resp_ind = np.argmax(Qvec)
            else:
                P_vec = P_vec/np.sum(P_vec)
                resp_ind = np.random.choice(np.arange(len(np.squeeze(Qvec))),1,p=np.squeeze(P_vec)).item()
        
        return resp_ind, P_vec
        
    def update_weights(self,RPE):
        self.W_r += self.beta*RPE*self.Tag_w_r
        self.V_r += self.beta*RPE*self.Tag_v_r

        self.W_m += self.beta*RPE*self.Tag_w_m
        self.V_m += self.beta*RPE*self.Tag_v_m
        if self.prop=='std' or self.prop=='BP':
            self.W_r_back += self.beta*RPE*np.transpose(self.Tag_w_r)
            self.W_m_back += self.beta*RPE*np.transpose(self.Tag_w_m)


    def reset_memory(self):
        self.cumulative_memory = 1e-6*np.ones((1,self.M))


    def reset_tags(self):

        self.sTRACE = 1e-6*np.ones((2*self.S, self.M))

        self.Tag_v_r = 1e-6*np.ones((self.S, self.R))
        self.Tag_v_m = 1e-6*np.ones((2*self.S, self.M))

        self.Tag_w_r = 1e-6*np.ones((self.R, self.A))
        self.Tag_w_m = 1e-6*np.ones((self.M, self.A))


    def update_tags(self,s_inst,s_trans,y_r,y_m,z,resp_ind):

        if isinstance(self.memory_leak, list):
            num_groups=int(self.M/len(self.memory_leak))
            leak_vec = np.repeat(self.memory_leak, num_groups)
            self.sTRACE = leak_vec*self.sTRACE + np.tile(np.transpose(s_trans), (1,self.M))
        else:
            self.sTRACE = self.memory_leak*self.sTRACE + np.tile(np.transpose(s_trans), (1,self.M))

        # synaptic tags for W
        self.Tag_w_r += -self.alpha*self.Tag_w_r + np.dot(np.transpose(y_r), z)
        delta_r = self.W_r_back[resp_ind,:]
        self.Tag_w_m += -self.alpha*self.Tag_w_m + np.dot(np.transpose(y_m), z)
        delta_m = self.W_m_back[resp_ind,:]

        # synaptic tags for V using feedback propagation and synaptic traces
        self.Tag_v_r += -self.alpha*self.Tag_v_r + np.dot(np.transpose(s_inst), y_r*(1-y_r)*delta_r)
        self.Tag_v_m += -self.alpha*self.Tag_v_m + self.sTRACE*y_m*(1-y_m)*delta_m

    def sigmoid(self,inp,W):
	    tot = np.dot(inp,W)
	    tot = np.clip(a=tot,a_min=-100,a_max=None) 
	    f = 1/(1+np.exp(-tot))
	    return f

    def sigmoid_acc_leaky(self, inp, W, acc, leak, gate=1):
    
    	if isinstance(leak, list):
    		num_groups=int(np.shape(acc)[1]/len(leak))
    		leak_vec = np.repeat(leak, num_groups)
    		tot1 = leak_vec*acc + gate*np.dot(inp,W)
    	else:
    		tot1 = leak*acc + gate*np.dot(inp,W)
    	tot2 = np.clip(a=tot1,a_min=-100,a_max=None) 
    	f = 1/(1+np.exp(-tot2))
    	return f,tot1

    def feedforward(self,s_inst,s_trans):

        y_r = self.sigmoid(s_inst, self.V_r)
        y_m,self.cumulative_memory = self.sigmoid_acc_leaky(s_trans, self.V_m, self.cumulative_memory,self.memory_leak)

        y_tot = np.concatenate((y_r, y_m),axis=1)
        W_tot = np.concatenate((self.W_r, self.W_m),axis=0)
        Q = np.dot(y_tot, W_tot)

        return y_r, y_m, Q
        

    def define_transient(self, s,s_old):

        s_plus =  np.where(s<=s_old,0,1)
        s_minus = np.where(s_old<=s,0,1)
        s_trans = np.concatenate((s_plus,s_minus),axis=1)

        return s_trans

    def learn(self,obs,target_act):

        s = np.zeros((1, self.S))
        s[0, obs] = 1
        o = np.zeros((1, self.A))
        o[0, target_act] = 1
                
        q = self.Q[0,self.resp_ind]
        
        z = np.zeros(np.shape(self.Q))
        z[0,self.resp_ind] = 1
                
        if self.first == False:
            RPE = (self.r + self.discount*q) - self.q_old  # Reward Prediction Error
            self.update_weights(RPE)
        elif self.first == True:
            self.first = False
        
        self.update_tags(self.s_inst,self.s_trans,self.y_r,self.y_m,z,self.resp_ind) 

        if self.resp_ind == target_act:
            self.r = self.rew_pos
        else: 
            self.r = self.rew_neg

        self.q_old = q
            
        RPE = self.r - self.q_old
        self.update_weights(RPE)

    def get_action(self, obs, episode_num, policy='eps_greedy', stoc='unif', t_weighted=False, e_weighted=False):

        s = np.zeros((1, self.S))
        s[0, obs] = 1

        self.s_inst = s
        self.s_trans = self.define_transient(self.s_inst, self.s_old)
        self.s_old = self.s_inst
                            
        self.y_r,self.y_m,self.Q = self.feedforward(self.s_inst, self.s_trans)
                
        self.resp_ind,_ = self.compute_response(self.Q, policy, stoc, t_weighted, e_weighted, episode_num)
                
        return self.resp_ind