class agent():
    
    def __init__(self):
        ###
        # This is just an example, please erase this line.
        self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        ###

    def load_weights(self):
        pass

    def action(self, state):
        ###
        # This is just an example, please erase this line.
        a = self.sample_actions.pop(0)
        self.sample_actions.append(a)
        ###
        return a