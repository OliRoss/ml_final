import gym
import random
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make('LunarLanderContinuous-v2')  

# Define hyperparameters
RANDOM_SEED = 123
NUM_EPISODES = 300

# Set seeds for reproducability
random.seed(RANDOM_SEED)  
env.seed(RANDOM_SEED)  
np.random.seed(RANDOM_SEED)

def get_action(observation):
    '''
    Function that takes random actions
    Arguments:
        observation: The state of the environment (array of 8 floats)
    Returns:
        randomly chosen action (array of 2 floats)
    '''
    return env.action_space.sample()

rewards = []
running_rewards = list()
running_reward = 1

for episode in range(NUM_EPISODES):
    observation = env.reset()
    episode_reward = 0
    while True:
        action = get_action(observation)
        observation, reward, done, info = env.step(action)
        # You can comment the below line for faster execution
        # env.render()
        episode_reward += reward
        if done:
            print('Episode: {} Reward: {}'.format(episode, episode_reward))
            rewards.append(episode_reward)
            break
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    running_rewards.append(running_reward)

plt.rcParams.update({'font.size': 18})

hp = {'name': 'random'}
label_str = hp['name'] 
plt.plot(range(len(running_rewards)), running_rewards, lw=2, color=np.random.rand(3, ), label=label_str)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Running average of Rewards')
plt.legend()
plt.show()

print('Average reward: %.2f' % (sum(rewards) / len(rewards)))

