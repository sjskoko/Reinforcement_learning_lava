# test_grid_search.py + save model weights

import random
import numpy as np
import torch
import argparse

import os
import sys


def evaluate_performance(team_number, seeds, env_str, env_kwargs):
    
    episodes = 50

    pf_list = []

    for seed in seeds:
        print(f'Seed {seed} start...')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = Env(**env_kwargs)

        agent_instance = Agent()
        agent_instance.load_weights() 

        pf = calculate_pf(episodes, env, agent_instance)
        pf_list.append(pf)
    
    print(f'Avg Performance: {np.mean(pf_list)}')
    
    with open(f"{cur_abs}/{env_str}-pf.txt", "a") as f:
        f.write(f"Team{team_number}:{np.mean(pf_list)}")


def evaluate_sample_efficiency(team_number, seeds, env_str, env_kwargs):
    
    if env_str == 'chain':
        episodes = 1000
    elif env_str == 'lava':
        episodes = 3000

    gamma_list = [0.95]
    lr_list = [0.005]
    noise1_list = [100.0]
    noise2_list = [5.0, None]

    idx = 0    
    for gamma in gamma_list:
        for learning_rate in lr_list:
            for noise1 in noise1_list:
                for noise2 in noise2_list:
                    se_list = []
                    idx += 1

                    for seed in seeds:
                        # print(f'Seed {seed} start...')
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)

                        env = Env(**env_kwargs)
                        agent_instance = Agent(learning_rate=learning_rate, gamma=gamma, noise1=noise1, noise2=noise2)

                        se = calculate_se(episodes, env, agent_instance)
                        se_list.append(round(se,2))

                        # save model weights
                        torch.save(agent_instance.model.state_dict(), f'model/DQN_{learning_rate}_{gamma}_{noise1}_{noise2}_{seed}.pt') # 경로에 폴더 하나 넣으면 에러 FileNotFoundError 뜸
                        print(f'Model saved: DQN_{learning_rate}_{gamma}_{noise1}_{noise2}_{seed}.pt')
    
                    print(f'({idx}) lr:{learning_rate} / gamma:{gamma} / noise1:{noise1} / noise2:{noise2} / {se_list} / Avg sample efficiency score : {np.mean(se_list)}')

    # with open(f"{cur_abs}/{env_str}-se.txt", "a") as f:
    #     f.write(f"Team{team_number}:{np.mean(se_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--team', required=True, type=int, choices=range(0,18),
                   metavar="[0-17]", 
                   help='team number (0 is for an example)')
    parser.add_argument('--envType', required=True, type=int, choices=range(0,2),
                   metavar="[0-1]", 
                   help='0: chain mdp 1: lava grid')
    parser.add_argument('--evalType', required=True, type=int, choices=range(0,3), default=0,
                   metavar="[0-2]", 
                   help='0: performance, 1: sample efficiency, 2: adaptability')
    parser.add_argument('--seeds', required=True, nargs='+', type=int, default=1)
    args = parser.parse_args()

    team_number = args.team
    seeds = args.seeds

    envType = args.envType
    
    global Agent, Env, calculate_pf, calculate_se

    if envType == 0:
        Agent = __import__(f'team{team_number}.agent_chainMDP', fromlist=['agent']).agent
        Env = __import__(f'chain_mdp', fromlist=['ChainMDP']).ChainMDP
        calculate_pf = __import__(f'team{team_number}.interaction_chainMDP', fromlist=['calculate_performance']).calculate_performance
        calculate_se = __import__(f'team{team_number}.interaction_chainMDP', fromlist=['calculate_sample_efficiency']).calculate_sample_efficiency

        env_str = 'chain'
        env_kwargs = {"n":10}

    elif envType == 1:
        Agent = __import__(f'team{team_number}.agent_lava', fromlist=['agent']).agent
        Env = __import__(f'lava_grid', fromlist=['ZigZag6x10']).ZigZag6x10
        calculate_pf = __import__(f'team{team_number}.interaction_lava', fromlist=['calculate_performance']).calculate_performance
        calculate_se = __import__(f'team{team_number}.interaction_lava', fromlist=['calculate_sample_efficiency']).calculate_sample_efficiency

        env_str = 'lava'
        env_kwargs = {"max_steps":100, "act_fail_prob":0, "goal":(5, 9), "numpy_state":False}

    ### Change Current directory for specific team directory
    global cur_abs
    cur_abs = os.path.abspath('.')
    os.chdir(f'{cur_abs}/team{team_number}')
    
    if args.evalType == 0:
        evaluate_performance(team_number, seeds, env_str, env_kwargs)
    elif args.evalType == 1:
        evaluate_sample_efficiency(team_number, seeds, env_str, env_kwargs)

