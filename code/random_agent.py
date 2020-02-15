import random
import numpy as np
import gym

# Define hyperparameters
RANDOM_SEED = 123

# Set seeds for reproducability
random.seed(RANDOM_SEED)  
np.random.seed(RANDOM_SEED)

class random_policy():
    def __init__(self):
        self.env = gym.make('LunarLanderContinuous-v2')

    def get_action(self):
        '''
        Function that takes random actions
        Arguments:
            observation: The state of the environment (array of 8 floats)
        Returns:
            randomly chosen action (array of 2 floats)
        '''
        return self.env.action_space.sample()

