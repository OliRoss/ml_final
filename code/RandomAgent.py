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

    def get_action(self,observation):
        '''
        Function that takes random actions
        Arguments:
            observation: The state of the environment (array of 8 floats)
        Returns:
            randomly chosen action (array of 2 floats)
        '''
        return self.env.action_space.sample()

    def evaluate(self,num_episodes):
        rewards = []
        i = 0

        for episode in range(num_episodes):
            if i % 10 == 0:
                print("Evaluating episode # {}".format(i))
            i = i + 1
            observation = self.env.reset()
            episode_reward = 0
            while True:
                action = self.get_action(observation)
                observation, reward, done, info = self.env.step(action)
                # You can comment the below line for faster execution
                # env.render()
                episode_reward += reward
                if done:
                    rewards.append(episode_reward)
                    break
        return rewards
