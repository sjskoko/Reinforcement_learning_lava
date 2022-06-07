import numpy as np
import random
import math

'''
수정중입니다
기존 q learning 알고리즘에서 n step sarsa로 변경 예정에 있으며
re-training code 역시 추가 필요합니다.
'''
class agent():
    
    def __init__(self, grid_size=60, epsilon=0.1, learning_rate = 0.1, gamma=1):

        self.q_table = dict()
        for x in range(grid_size):
            self.q_table[x] = {0:0, 1:0, 2:0, 3:0} # 0: left, 1: up, 2: right, 3: down


        self.action_space = [0, 1, 2, 3]

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.episode_num = 3000
        
    def action(self, s, greedy = False):

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

    def activate_learn(self, num_episodes):
        
