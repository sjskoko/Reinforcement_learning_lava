import numpy as np

def calculate_performance(episodes, env, agent):

    episodic_returns = []
    
    for epi in range(episodes):
        
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s, epi) # 수정
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            s = ns
        
        episodic_returns.append(cum_reward)

    # 추가
    print('max reward:', max(episodic_returns))

    import matplotlib.pyplot as plt
    x = [i+1 for i in range(len(episodic_returns))]
    plt.plot(x, episodic_returns)
    plt.title('Episode Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('test_pf.png')
    plt.close()    
    
    return np.sum(episodic_returns)

def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = []
    
    for epi in range(episodes):
        
        s = env.reset() # s: numpy.int64

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s, epi) # 수정
            ns, reward, done, _ = env.step(action) # ns: np.array
            cum_reward += reward
            agent.update(ns, action, reward, epi) # 수정
            s = ns
        
        episodic_returns.append(cum_reward)

    # 추가
    print('max reward:', max(episodic_returns))

    import matplotlib.pyplot as plt
    x = [i+1 for i in range(len(episodic_returns))]
    plt.plot(x, episodic_returns)
    plt.title('Episode Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('test_se.png')
    plt.close()        
                    
    return np.sum(episodic_returns)

