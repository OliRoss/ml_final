import numpy as np
import gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from linearFAPolicyAgent import LinearFAPolicy

import itertools
from itertools import count
import matplotlib.pyplot as plt
from tqdm import trange

# Initialize the environment
# env = gym.make('LunarLanderContinuous-v2')
env = gym.make('CartPole-v1')

# Define hyperparameters
RANDOM_SEED = 123
NUM_EPISODES = 50

# Set seeds for reproducability
random.seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def poly_features(state, n):
    '''
    Computes the polynomial feature vector from the input states

    :param: state: Input state vector of size k (1D numpy array)
    :param: n: polynomial degree
    :return: feature vector phi consisting of (n+1)^k elements (1D numpy array)
    '''

    k = state.shape[0]
    phi = np.zeros((n + 1) ** k)

    # Your code starts here

    # calculate the c-vectors from Sutton and Burato p. 211
    c = np.zeros(((n + 1) ** k, k))
    num = 0
    for pos in range(k):
        for row in range((n + 1) ** k):
            c[row][pos] = num
            if (row + 1) % ((n + 1) ** pos) == 0:
                if num < n:
                    num += 1
                else:
                    num = 0

    # calculate the feature vector phi
    for i in range(len(phi)):
        phi[i] = 1
        for j in range(k):
            phi[i] *= (state[j] ** c[i][j])

    # Your code ends here

    return phi


def finish_episode():
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

    # Here, we deviate from the standard REINFORCE algorithm as discussed above
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


def reinforce(render=False, log_interval=100, max_episodes=3000):
    # To track the reward across consecutive episodes (smoothed)
    running_reward = 1.0

    # Lists to store the episodic and running rewards for plotting
    ep_rewards = list()
    running_rewards = list()

    # Start executing an episode (here the number of episodes is unlimited)
    for i_episode in count(1):

        # Reset the environment
        state, ep_reward = env.reset(), 0

        # For each step of the episode
        for t in range(1, 10000):

            # Select an action using the policy network
            action = policy.select_action(state)

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
        finish_episode()

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        # Stopping criteria
        if running_reward > env.spec.reward_threshold:
            print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
            break
        if i_episode >= max_episodes:
            print('Max episodes exceeded, quitting.')
            break
    # Save the trained policy network
    policy.save()

    return ep_rewards, running_rewards


# A function for training the linear FA policy parameters with specified hyperparameters
def train_policy(**hyperparam_dict):
    # Fetch the hyperparameters
    global gamma, learning_rate, poly_degree
    gamma = hyperparam_dict['gamma']
    learning_rate = hyperparam_dict['learning_rate']
    poly_degree = hyperparam_dict['poly_degree']

    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    feature_size = (poly_degree + 1) ** state_size

    # Create the policy function and set the training mode
    global policy
    policy = LinearFAPolicy(feature_size=feature_size, num_actions=num_actions, feature_func=poly_features,
                            poly_degree=poly_degree)
    policy.train()

    # Define the optimizer and set the learning rate
    global optimizer
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Execute the REINFORCE algorithm with linear function ap
    ep_rewards, running_rewards = reinforce()

    del optimizer
    del policy

    return ep_rewards, running_rewards
