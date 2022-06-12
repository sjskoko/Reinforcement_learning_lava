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

    def n_step_sarsa(self, num_episodes, n=5, discount_factor=1.0, alpha=0.5):

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

        
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # Keeps track of useful statistics
        stats = plots.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        # The policy we're following
        
        for i_episode in range(num_episodes):

            self.episode_num += 1

            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 10 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            # initializations
            T = sys.maxsize
            tau = 0
            t = -1
            stored_actions = {}
            stored_rewards = {}
            stored_states = {}
            
            # initialize first state
            state = env.reset()
            if self.episode_num<=3000:
                action = self.action(state, greedy=False)
            else:
                action = self.action(state, greedy=True)      
                  
            stored_actions[0] = action
            stored_states[0] = state
            
            while tau < (T - 1):
                t += 1
                if t < T:
                    state, reward, done, _ = env.step(action)

                    state = self.array_to_index(state)
                    
                    stored_rewards[(t+1) % (n+1)] = reward
                    stored_states[(t+1) % (n+1)] = state
                    
                    # Update statistics
                    stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t
                    
                    if done:
                        T = t + 1
                    else:
                        if self.episode_num<=3000:
                            action = self.action(state, greedy=False)
                        else:
                            action = self.action(state, greedy=True)   
                        stored_actions[(t+1) % (n+1)] = action
                tau = t - n + 1
                
                if tau >= 0:
                    # calculate G(tau:tau+n)
                    G = np.sum([discount_factor**(i-tau-1)*stored_rewards[i%(n+1)] for i in range(tau+1, min(tau+n, T)+1)])
                    
                    
                    if tau + n < T:
                        # G += discount_factor**n * self.q_table[stored_states[(tau+n) % (n+1)]][stored_actions[(tau+n) % (n+1)]]
                        # expected sarsa
                        G += discount_factor**n * self.expected_value(stored_states[(tau+n) % (n+1)])
                    
                    tau_s, tau_a = stored_states[tau % (n+1)], stored_actions[tau % (n+1)]
                    
                    # update Q value with n step return
                    self.q_table[tau_s][tau_a] += alpha * (G - self.q_table[tau_s][tau_a])
            
        return self.q_table, stats