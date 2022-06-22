import numpy as np
from tqdm import tqdm

def calculate_performance(episodes, env, agent):

    episodic_returns = []
    
    for epi in range(episodes):
        
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            s = ns
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = []
    
    for epi in tqdm(range(episodes)):
        
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            
            #####################
            # If your agent needs to update the weights at every time step, complete your update process in this area.
            agent.observe(s, action, ns, reward, done)
            agent.update_after_step()

            #####################
            s = ns

        #####################
        # elif your agent needs to update the weights at the end of every episode, complete your update process in this area.
        # e.g., agent.update()


        #####################
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

