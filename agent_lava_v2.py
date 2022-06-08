import numpy as np
import random
import math

import gym
from lava_grid import ZigZag6x10


'''
수정중입니다
주어진 update test 환경에서 구동가능하도록 수정 완료했으며
re-training code 추가가 필요합니다.
또한 default parameter로 기존 q-table 변경 필요합니다.'''

class agent():
    
    def __init__(self, grid_size=60, epsilon=0.1, learning_rate = 0.1, gamma=1):

        self.grid_size = grid_size
        self.q_table = dict()


        self.action_space = [0, 1, 2, 3]

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.episode_num = 3000
        
        self.activate_learn(3000) # initial parameter setting
        
    def action(self, s, greedy = True):

        if type(s) == np.ndarray:
            state_index = np.where(s==1)[0][0]
            s=state_index

        if np.random.uniform(0,1) < self.epsilon/(math.log10(self.episode_num)+1) and not greedy:
            action = random.randint(0, len(self.action_space)-1)
        else:
            q_values_of_state = self.q_table[s]
            max_value = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == max_value])

        
        return action

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
                action = self.action(s, greedy=False)
                ns, reward, done, _ = env.step(action)
                ns = env.s
                self.learn(s, reward, ns, action)
                cum_reward += reward
                s = ns
            
            record.append(cum_reward)
        
        return record
            





'''
본 파일은 
V2
입니다.'''