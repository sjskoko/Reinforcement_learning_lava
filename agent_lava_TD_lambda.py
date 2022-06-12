from typing import DefaultDict
import numpy as np
import random
import math
from tqdm import tqdm
from collections import defaultdict
import sys
from lib import plots

import gym
from lava_grid import ZigZag6x10


class agent():
    
    def __init__(self, grid_size=60, epsilon=0, learning_rate = 0.1, gamma=1, set_parameter=True):

        self.grid_size = grid_size
        self.q_table = dict()
        for x in range(self.grid_size):
            self.q_table[x] = {0:0, 1:0, 2:0, 3:0} # 0: left, 1: up, 2: right, 3: down

        self.v_table = dict()
        for x in range(self.grid_size):
            self.v_table[x] = 0.0
        


        self.action_space = [0, 1, 2, 3]

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.episode_num = 3000
        
        if set_parameter==True:
            self.activate_learn(3000) # initial parameter setting
        
    # Take action based on the epsilon
    def action(self, state, greedy = True):

        if type(state) == np.ndarray:
            state_index = np.where(state==1)[0][0]
            state=state_index

        # if np.random.uniform(0,1) < self.epsilon/(math.log10(self.episode_num)+1) and not greedy:
        if np.random.uniform(0,1) < self.epsilon and not greedy:
            action = random.randint(0, len(self.action_space)-1)
        else:
            q_values_of_state = self.q_table[state]
            max_value = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == max_value])
        
    def action_v(self, state, greedy = True):
        
        state = self.array_to_index(state)

        # if np.random.uniform(0,1) < self.epsilon/(math.log10(self.episode_num)+1) and not greedy:
        if np.random.uniform(0,1) < self.epsilon and not greedy:
            action = random.randint(0, len(self.action_space)-1)
        else:
            state_coordinate = [state//10, state%10] # row, colunm
            state_val = self.v_table[state]
            cand_state_dict = {}
            for i in self.action_space:
                if i == 0:
                    if state_coordinate[1]-1 >= 0:
                        cand_state_dict[i] = 10*state_coordinate[0] + state_coordinate[1]
                elif i == 1:
                    if state_coordinate[0]-1 >= 0:
                        cand_state_dict[i] = 10*state_coordinate[0] + state_coordinate[1]
                
                elif i == 2:
                    if state_coordinate[1]+1 <= 9:
                        cand_state_dict[i] = 10*state_coordinate[0] + state_coordinate[1]
                
                elif i == 3:
                    if state_coordinate[0]+1 <= 9:
                        cand_state_dict[i] = 10*state_coordinate[0] + state_coordinate[1]


            max_value = max(cand_state_dict.values())
            action = np.random.choice([k for k, v in cand_state_dict.items() if v == max_value])

        return action
    
    def array_to_index(self, state):
        if type(state) == np.ndarray:
            state_index = np.where(state==1)[0][0]
            state=state_index
        return state

    # Calculate the expected value for each action based on the epsilon 
    def expected_value(self, state):
        
        q_values_of_state = self.q_table[state]
        max_value = max(q_values_of_state.values())
        max_value_action = [k for k, v in q_values_of_state.items() if v == max_value]

        prob_dict = {}
        for action in q_values_of_state:
            epsilon_value = self.epsilon/(math.log10(self.episode_num)+1)
            prob_dict[action] = epsilon_value/len(q_values_of_state)
            if action in max_value_action:
                prob_dict[action] += (1-epsilon_value)/len(max_value_action)
        
        exp_val_dict = {}
        for action in q_values_of_state:
            exp_val_dict[action] = prob_dict[action] * q_values_of_state[action]

        sum_exp_val = 0
        for action in exp_val_dict:
            sum_exp_val += exp_val_dict[action]
        
        return sum_exp_val

    def learn(self, old_state, reward, new_state, action):

        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]

        self.q_table[old_state][action] = (1-self.learning_rate) * current_q_value + self.learning_rate * (reward + self.gamma * max_q_value_in_new_state)

    def reset_q_table(self):
        # reset q_table
        self.q_table = dict()
        for x in range(self.grid_size):
            self.q_table[x] = {0:0, 1:0, 2:0, 3:0} # 0: left, 1: up, 2: right, 3: down


    def td_lambda(self, num_episodes, discount_factor=1.0, alpha=0.5, lambd=0.1):

        # reset episode_num
        self.episode_num = 0

        # reset q_table
        self.q_table = dict()
        for x in range(self.grid_size):
            self.q_table[x] = {0:0, 1:0, 2:0, 3:0} # 0: left, 1: up, 2: right, 3: down

        '''setting environment'''
        # default setting
        max_steps = 100
        stochasticity = 0 # probability of the selected action failing and instead executing any of the remaining 3
        no_render = True

        env = ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)
        done = False
        cum_reward = 0.0

        
        # The final value function
        
        # Implement this!
        for episode in range(num_episodes):

            self.episode_num += 1

            E = defaultdict(float)
            state = env.reset()
            
            for t in range(1000):
                print(state)
                # sample action from the policy
                action = self.action_v(state)
                
                # environments' effects after taking action
                next_state, reward, done, _ = env.step(action)
                next_state = self.array_to_index(next_state)
                
                # update eligibility trace
                E[state] += 1
                
                td_error = reward + (discount_factor * self.v_table[next_state]) - self.v_table[state]
                
                
                # online update value function
                for s, value in self.v_table.items():
                    self.v_table[s] = self.v_table[s] + alpha * td_error * E[s]
                    E[s] = discount_factor * lambd * E[s]
                if done:
                    break
                    
                state = next_state

        return self.v_table