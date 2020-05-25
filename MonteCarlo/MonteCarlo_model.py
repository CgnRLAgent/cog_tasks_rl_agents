import random
from random import randint
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.agent import Agent

class Agent_MC(Agent):

    def __init__(self, n_obs, n_act):
        super().__init__(n_obs, n_act)
        self.history = {}
        self.is_training = False
        self.pastAction = []
        self.totalRewards = 0
        self. n_act = n_act
        #self.maxValueAction = None
        self.randomAction = None
        self.i = 1
        

    def train(self):

        self.is_training = True
        return

    def eval(self):

        self.is_training = False
        return

    def reset(self):

        self.pastAction = []
        self.totalRewards = 0

    def respond(self, obs):

        if self.is_training:
            if obs not in self.history: 
                self.history[obs] = []
            # Get the most valued action
            maxValueAction = None
            for a in self.history[obs]:
                if maxValueAction == None or a['value'] > maxValueAction['value']:
                    maxValueAction = a
                    #print(maxValueAction)
            # If no maxValueAction or if it's time for exploration, return random action
            if maxValueAction == None or randint(0, 100) < 20*0.99**self.i:
                
                self.randomAction = {'value': 0, 'obs':obs, 'action': np.random.choice(self.n_act), 'count': 0}
                
                act = self.actionExists(obs, self.randomAction)
                if act is None:
                    act = self.randomAction
                    self.history[obs].append(self.randomAction)

                self.i+=1
                self.pastAction.append(act)
                return act['action']
            else:
                self.i+=1
                self.pastAction.append(maxValueAction)
                return maxValueAction['action']
            

        else:
            policy = {}
            #print(self.history)
            for ob in self.history:
                #print(ob)
                maxValueAction = None
                for a in self.history[ob]:
                    
                    if maxValueAction == None or a['value'] > maxValueAction['value']:
                        maxValueAction = a
            #print(self.maxValueAction)
                policy[ob] = maxValueAction['action']
            #print(policy)
            return policy[obs]

    def learn(self, obs, next_obs, action, reward, done, target_act=None):
        """
        after responding to an observation, agent can learn from the rewards or golden action(if has).
        :param obs: (int)
        :param action: the agent's action reponding to the observation. (int)
        :param reward:
        :param done: if the episode is end after taking the action.
        :param target_act: the golden action responding to the observation.

        """
        #self.pastAction.append(action)
        pastAction = self.pastAction
        self.totalRewards += reward
        if done:
            for i, a in enumerate(pastAction):
                self.updatePolicy(a, self.totalRewards)
        
    def updatePolicy(self, action, G):
        action['value'] = (action['value'] * action['count'] + G) / (action['count'] + 1)
        action['count'] += 1
    
    def actionExists(self, obs, action):
        for a in self.history[obs]:
            if self.actionAreEqual(a['action'], action['action']):
                return a
        return None

    def actionAreEqual(self, action1, action2):
        return action1 == action2

    def save(self, dir, name):
        """
        save the agent's parameters and configs to the specified path
        2 saved files, ${name}_params, ${name}_configs
        :param dir: (str)
        :param name: agent's name
        """
        return

    def load(self, path):
        """
        load pre-trained parameters of the agent
        :param path: (str)
        """
        return