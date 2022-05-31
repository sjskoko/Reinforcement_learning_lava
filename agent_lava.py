import numpy as np

class agent():
    
    def __init__(self, grid_size=(4, 60), learning_rate = 0.9, gamma=0.9):
        # self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        self.action_space = [0, 1, 2, 3]
        self.state_value = self._state_value_function(grid_size)
        self.grid_size = grid_size
        self.step = 0
        self.learning_rate = learning_rate
        self.gamma = gamma
        
    def action(self):
        
        return self.sample_actions.pop(0)

    def _state_value_function(self, grid_size):
        state_value = np.zeros(grid_size)

        return state_value
    
    # def _max_Q(self, state):
    #     state_index = int(np.where(state==1))
    #     target_indexs = [state_index-1, state_index-10, state_index+1, state_index+10]
    #     max_q = 0
    #     action = None
    #     for i, target_index in enumerate(target_indexs):
    #         if target_index>=0 and target_index<60:
    #             cand_q = self.state_value[target_index]
    #             if cand_q>max_q:
    #                 max_q = cand_q
    #                 action = i
    #     return max_q
    
    # def _argmax_Q(self, state):
    #     state_index = int(np.where(state==1))
    #     target_indexs = [state_index-1, state_index-10, state_index+1, state_index+10]
    #     max_q = 0
    #     action_q_pair = []
    #     for i, target_index in enumerate(target_indexs):
    #         if target_index>=0 and target_index<60:
    #             cand_q = self.state_value[target_index]
    #             action_q_pair.append((i, cand_q))
    #     return action_q_pair

    # complete
    def _max_Q(self, state):
        state_index = np.where(state==1)[0]
        target_indexs = [state_index-1, state_index-10, state_index+1, state_index+10]
        cand_q = []
        for i, target_index in enumerate(target_indexs):
            if target_index>=0 and target_index<60:
                cand_q.append(self.state_value[:, target_index])
        max_q = max([max(i) for i in cand_q])

        return max_q[0]

    # 수정중
    def _argmax_Q(self, state):
        state_index = np.where(state==1)[0]
        target_indexs = [state_index-1, state_index-10, state_index+1, state_index+10]
        cand_q = []
        for i, target_index in enumerate(target_indexs):
            if target_index>=0 and target_index<60:
                cand_q.append(self.state_value[:, target_index])
        max_q = max([max(i) for i in cand_q])

        return max_q[0]

    # 수정중
    def state_value_update(self, state, action, reword):
        state_index = int(np.where(state==1))
        self.state_value[action, state_index] += self.learning_rate*(reword + self.gamma*self._max_Q(state) - self.state_value[action, state_index])






        

if __name__ == '__main__':
    agent = agent()

    temp = (agent.state_value)