def n_step_sarsa(env, gamma, n_episode, alpha , n , learn_pi = True):
    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    
    policy = {}
    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        action = epsilon_greedy_policy(state, Q)
        print(Q)

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
                next_action = epsilon_greedy_policy(next_state, Q)
                n_step_store["states"].append(next_state)
                n_step_store["actions"].append(next_action)
                n_step_store["rewards"].append(reward)
                if is_done :
                    total_reward_episode[episode] += np.sum(n_step_store["rewards"])
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
                    G += (gamma ** n) * Q[n_step_store["states"][tau + n]][n_step_store["actions"][tau + n]]
                    
                Q[n_step_store["states"][tau]][n_step_store["actions"][tau]] += alpha * (G - Q[n_step_store["states"][tau]][n_step_store["actions"][tau]])
                ## On-Policy 바로 바로 학습 n-step 이후로는 바로 학습하는 구조인 듯
                if learn_pi :    
                    policy[n_step_store["states"][tau]] = epsilon_greedy_policy(n_step_store["states"][tau], Q)
            state = next_state
            action = next_action
            if tau == (T-1):
                    break
            t += 1
    return Q, policy

gamma = 1
n_episode = 500
alpha = 0.4
epsilon = 0.1
epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)
length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode