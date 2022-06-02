'''
https://github.com/michaeltinsley/Gridworld-with-Q-Learning-Reinforcement-Learning-/blob/master/Gridworld.ipynb

위 repo의 내용을 토대로 작성중
'''


import numpy as np
import random

class agent():
    
    def __init__(self, environment, epsilon=0.05, learning_rate = 0.1, gamma=1):

        self.environment = environment
        self.q_table = dict()
        for x in range(environment.nS):
            self.q_table[x] = {0:0, 1:0, 2:0, 3:0} # 0: left, 1: up, 2: right, 3: down


        self.action_space = [0, 1, 2, 3]

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        
    def action(self):
        
        if np.random.uniform(0,1) < self.epsilon:
            action = random.randint(0, len(self.action_space)-1)
        else:
            q_values_of_state = self.q_table[self.environment.s]
            max_value = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == max_value])
        return action

    def learn(self, old_state, reward, new_state, action):

        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]

        self.q_table[old_state][action] = (1-self.learning_rate) * current_q_value + self.learning_rate * (reward + self.gamma * max_q_value_in_new_state)





        

if __name__ == '__main__':
    agent = agent()

    temp = (agent.state_value)



    # https://github.com/michaeltinsley/Gridworld-with-Q-Learning-Reinforcement-Learning-/blob/master/Gridworld.ipynb