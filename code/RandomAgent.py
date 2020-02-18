import random
import numpy as np
import gym

# Define hyperparameters
RANDOM_SEED = 123

# Set seeds for reproducability
random.seed(RANDOM_SEED)  
np.random.seed(RANDOM_SEED)


class random_policy:
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
        '''
        Function that evaluates the random agent's performance
        :param num_episodes: How many episodes the agent will be tested for
        :return: list that contains the rewards achieved per episode
        '''

        rewards = []
        i = 0

        # Iterate through episodes
        for episode in range(num_episodes):

            # Track progress on stdout every 10 episodes
            if i % 10 == 0:
                print("Evaluating episode # {}".format(i))
            i = i + 1

            # Get inital state from environment
            observation = self.env.reset()
            episode_reward = 0

            # Iterate through the steps of the episode
            while True:
                # Get action
                action = self.get_action(observation)

                # Perform step and observe
                observation, reward, done, info = self.env.step(action)

                # Calculate episode reward
                episode_reward += reward

                if done:
                    rewards.append(episode_reward)
                    break

        return rewards
