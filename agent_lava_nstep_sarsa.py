from typing import DefaultDict
import numpy as np
import random
import math
from tqdm import tqdm
from collections import defaultdict

import gym
from lava_grid import ZigZag6x10


class agent():
    
    def __init__(self, grid_size=60, epsilon=0, learning_rate = 0.1, gamma=1, set_parameter=True):

        self.grid_size = grid_size
        self.q_table = dict()
        for x in range(self.grid_size):
            self.q_table[x] = {0:0, 1:0, 2:0, 3:0} # 0: left, 1: up, 2: right, 3: down


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

        if np.random.uniform(0,1) < self.epsilon/(math.log10(self.episode_num)+1) and not greedy:
            action = random.randint(0, len(self.action_space)-1)
        else:
            q_values_of_state = self.q_table[state]
            max_value = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == max_value])

        
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

        exp_val_dict = {}
        for action in q_values_of_state:
            epsilon_value = self.epsilon/(math.log10(self.episode_num)+1)
            exp_val_dict[action] = epsilon_value/len(q_values_of_state)
            if action in max_value_action:
                exp_val_dict[action] += (1-epsilon_value)/len(max_value_action)
        
        return exp_val_dict

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

    def activate_learn(self, num_episodes):
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
        s = env.reset()
        done = False
        cum_reward = 0.0

        record = []

        # training based on given episode number
        for i_episode in tqdm(range(num_episodes)):
            self.episode_num += 1

            '''reset environment'''
            s = env.reset()
            done = False
            cum_reward = 0.0

            # 
            while not done:
                if self.episode_num<=300:
                    action = self.action(s, greedy=False)
                else:
                    action = self.action(s, greedy=True)
                ns, reward, done, _ = env.step(action)
                ns = env.s
                self.learn(s, reward, ns, action)
                cum_reward += reward
                s = ns
            
            record.append(cum_reward)
        
        return record

    def n_step_sarsa(self, gamma, n_episode, alpha , n , learn_pi = True):

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

        record = defaultdict(float)
        length_episode = defaultdict(int)
        
        policy = {}
        for episode in tqdm(range(n_episode)):
            self.episode_num += 1
            print('episode', episode)
            state = env.reset()
            is_done = False
            if self.episode_num<=3000:
                action = self.action(state, greedy=False)
            else:
                action = self.action(state, greedy=True)
            # print(action)
            s = ['states', 'actions', 'rewards']
            n_step_store = defaultdict(list)
            for key in s :
                n_step_store[key] 
            n_step_store["states"].append(state)
            n_step_store["actions"].append(action)
            n_step_store["rewards"].append(0)
            t, T = 0 , 10000
            while True :
                if t < T : 
                    next_state, reward, is_done, info = env.step(action)
                    next_state = self.array_to_index(next_state) # nd.array -> int
                    if self.episode_num<=3000:
                        next_action = self.action(next_state, greedy=False)
                    else:
                        next_action = self.action(next_state, greedy=True)

                    # print('in loop', next_action)
                    n_step_store["states"].append(next_state)
                    n_step_store["actions"].append(next_action)
                    n_step_store["rewards"].append(reward)
                    if is_done :
                        record[episode] += np.sum(n_step_store["rewards"])
                        T = t + 1
                    else :
                        length_episode[episode] += 1
                print(f"{t} / {T}" , end="\r")
                tau = t-n + 1
                if tau >= 0 :
                    G = 0 
                    ## G만 구하는 과정 (현재 시점)
                    for i in range(tau + 1, min([tau + n, T]) + 1):
                        G += (gamma ** (i - tau - 1)) * n_step_store["rewards"][i-1]
                    ## (미래 시점) 더하는 부분
                    if tau + n < T :
                        G += (gamma ** n) * self.q_table[n_step_store["states"][tau + n]][n_step_store["actions"][tau + n]]
                        
                    self.q_table[n_step_store["states"][tau]][n_step_store["actions"][tau]] += alpha * (G - self.q_table[n_step_store["states"][tau]][n_step_store["actions"][tau]])
                    ## On-Policy 바로 바로 학습 n-step 이후로는 바로 학습하는 구조인 듯
                    if learn_pi :    
                        policy[n_step_store["states"][tau]] = self.action(n_step_store["states"][tau], greedy=False)
                state = next_state
                action = next_action
                if tau == (T-1):
                    print(n_step_store)
                    break
                t += 1
        return policy, record