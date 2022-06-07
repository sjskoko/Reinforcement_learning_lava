import numpy as np
import gym
import random
from lava_grid import ZigZag6x10
from tqdm import tqdm
import math
from agent_lava_v2 import agent

import gym
from lava_grid import ZigZag6x10
from agent_lava_v2 import agent
import random

# default setting
max_steps = 100
stochasticity = 0 # probability of the selected action failing and instead executing any of the remaining 3
no_render = True

env = ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)
reward_record = []

for i in range(1000):

    s = env.reset()
    done = False
    cum_reward = 0.0

    """ Your agent"""
    my_agent = agent()
    my_agent.activate_learn(3000)

    # moving costs -0.01, falling in lava -1, reaching goal +1
    # final reward is number_of_steps / max_steps
    while not done:
        action = my_agent.action(s)
        # action = random.randrange(4): random actions
        ns, reward, done, _ = env.step(action)
        cum_reward += reward
        s = ns
    print(f"total reward: {cum_reward}")
    reward_record.append(cum_reward)