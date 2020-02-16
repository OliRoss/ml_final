import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributions as torch_dist
import torch.nn.functional as F
import torch.optim as optim

# Initialize the environment
env = gym.make('LunarLanderContinuous-v2')

# Define hyperparameters
RANDOM_SEED = 123
SAVE_INTERVAL = 1

# Set seeds for reproducability
random.seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def reinforce(policy, learning_rate=False,render=False, num_episodes=100, gamma=0.9, log_interval=1):

    policy.train()

    # To track the reward across consecutive episodes (smoothed)
    running_reward = -1250.0

    # Lists to store the episodic and running rewards for plotting
    ep_rewards = list()
    running_rewards = list()

    # Start executing an episode (here the number of episodes is unlimited)
    for i_episode in range(num_episodes):

        # Reset the environment
        state, ep_reward = env.reset(), 0

        # For each step of the episode
        for t in range(1, 1000):

            # Select an action using the policy network
            mu, action, log_prob = policy.gaussian_policy(state)

            # Perform the action and note the next state and reward
            state, reward, done, _ = env.step(action)

            if render:
                env.render()

            # Store the current reward
            policy.rewards.append(reward)

            # Track the total reward in this episode
            ep_reward += reward

            if done:
                break

        # Update the running reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Store the rewards for plotting
        ep_rewards.append(ep_reward)
        running_rewards.append(running_reward)

        # Perform the parameter update according to REINFORCE
        perform_update(policy,learning_rate,gamma)

        if i_episode % log_interval == 0:
            print(i_episode, ep_reward, running_reward)
        # Stopping criteria
        if i_episode % SAVE_INTERVAL == 0:
            policy.save()
        if running_reward > env.spec.reward_threshold:
            print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
            break


        # Plot the running average results
        fig = plt.figure(0, figsize=(20, 8))
        plt.rcParams.update({'font.size': 18})

        hp = {'name': 'Neural_net', 'gamma': gamma, 'learning_rate': learning_rate}
        label_str = hp['name'] + '(gamma:' + str(hp['gamma']) +  ',lr:' + str(
            hp['learning_rate']) + ')'
        file_str = label_str + datetime.now().strftime("_%d_%m_%H:%M") + '.png'
        plt.plot(range(len(running_rewards)), running_rewards, lw=2, color=np.random.rand(3, ), label=label_str)
        plt.grid()
        plt.xlabel('Episodes')
        plt.ylabel('Running average of Rewards')
        plt.legend()
        plt.savefig(file_str)
        plt.show()


    return ep_rewards, running_rewards

def perform_update(policy, learning_rate, gamma = 0.9):
    # Define the optimizer and set the learning rate
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Variable for the current return
    G = 0
    policy_loss = []
    returns = []

    # Define a small float which is used to avoid divison by zero
    eps = np.finfo(np.float32).eps.item()

    # Go through the list of observed rewards and calculate the returns
    for r in policy.rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)

    # Convert the list of returns into a torch tensor
    returns = torch.tensor(returns)

    # Here we normalize the returns by subtracting the mean and dividing
    # by the standard deviation. Normalization is a standard technique in
    # deep learning and it improves performance, as discussed in
    # http://karpathy.github.io/2016/05/31/rl/
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # Product of the corresponding logprobs and returns
    # This is used to compute the loss for backpropagation
    for log_prob, G in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * G)

    # Reset the gradients of the parameters
    optimizer.zero_grad()

    # Compute the cumulative loss
    policy_loss = torch.cat(policy_loss).mean()

    # Backpropagate the loss through the network
    policy_loss.backward()

    # Perform a parameter update step
    optimizer.step()

    # Reset the saved rewards and log probabilities
    del policy.rewards[:]
    del policy.saved_log_probs[:]

