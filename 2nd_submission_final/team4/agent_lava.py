import torch
import numpy as np
import random
import math

class agent():
    
    def __init__(self, nState=60, nAction=4, learning_rate=0.005, gamma=0.95, noise1=50.0, noise2=1):
        
        self.nS = nState
        self.nA = nAction
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.nS,self.nA,bias=False)
        )

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.noise1 = noise1
        self.noise2 = noise2
        
        self.loss = torch.nn.MSELoss()
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.step = 0
        self.qval_temp = None

    def load_weights(self):
        noise2_tmp = self.noise2
        if self.noise2 == None:
            noise2_tmp = 0
        # self.model.load_state_dict(torch.load(f'C:/Users/sjsko/PycharmProjects/Reinforcement_learning_lava/2nd_submission_final/models/{self.learning_rate}_{self.gamma}_{self.noise1}_{noise2_tmp}.pt'))
        self.model.load_state_dict(torch.load(f'C:/Users/sjsko/PycharmProjects/Reinforcement_learning_lava/2nd_submission_final/models/final_model.pt'))

    def action(self, state):
        self.step += 1

        if type(state) == np.int64 or type(state) == np.int32:
            state = np.eye(self.nS)[state]
        state = torch.from_numpy(state).float()

        self.qval_temp = self.model(state)
        qval_ = self.qval_temp.data.numpy()

        # add random noise to actions
        if self.noise1 != None:
            qval_ += np.random.rand(self.nA) * self.noise1/(self.step+1)

        a = np.argmax(qval_)

        return a

    def update(self, new_state, action, reward):
        new_state = torch.from_numpy(new_state).float() 

        with torch.no_grad():
            newQ = self.model(new_state)

        # add random noise to actions
        if self.noise2 != None:
            newQ += torch.randn(self.nA) * self.noise2/(self.step+1)

        maxQ = torch.max(newQ) 
        if reward == -0.01:
            Y = reward + (self.gamma * maxQ)
        else:
            Y = reward

        X = self.qval_temp.squeeze()[action].reshape([1])
        Y = torch.Tensor([Y]).detach()
        loss = self.loss(X, Y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
