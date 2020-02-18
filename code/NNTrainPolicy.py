import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.optim as optim

# Initialize the environment
env = gym.make('LunarLanderContinuous-v2')

# Define hyperparameters
SAVE_INTERVAL = 50


def reinforce(policy, learning_rate, render=False, num_episodes=100, gamma=0.9, log_interval=1):

    # Set neural network to training mode
    policy.train()

    # To track the reward across consecutive episodes
    running_reward = -250

    # To track the best running reward encountered so far
    best_running_reward = -250

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
            state, reward, done, _ = env.step(np.array([action[0][0], action[0][1]]))

            # Render, if necessary
            if render == "True":
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

        # Log to stdout
        if i_episode % log_interval == 0:
            print('Finished episode {}\tEpisode reward: {:.2f}\tAverage reward: {:.2f}'.format(i_episode, ep_reward, running_reward))

        # Save if running reward improved over all running rewards
        if running_reward > best_running_reward:
            policy.save(policy.file_name + '_best')
            best_running_reward = running_reward
            np.savetxt(policy.file_name + '_best_ep_rewards.csv', ep_rewards, delimiter=",")

        # Save after every 50 episodes
        if i_episode % SAVE_INTERVAL == 0:
            policy.save(policy.file_name + '_regular')
            np.savetxt(policy.file_name + '_regular_ep_rewards.csv', ep_rewards, delimiter=",")

        # Stopping criteria
        if running_reward > env.spec.reward_threshold:
            print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
            break

    # Save last policy, and the recorded rewards
    policy.save(policy.file_name + '_regular')
    np.savetxt(policy.file_name + '_regular_ep_rewards.csv', ep_rewards, delimiter=",")

    # Plot the running average results
    fig = plt.figure(0, figsize=(20, 8))
    plt.rcParams.update({'font.size': 18})
    hp = {'name': 'Neural_net', 'gamma': gamma, 'learning_rate': learning_rate, 'random_seed': policy.random_seed}
    label_str = hp['name'] + '(gamma:' + str(hp['gamma']) +  ',lr:' + str(
        hp['learning_rate']) + ',random seed: ' + str(hp['random_seed']) + ')'
    file_str = label_str + datetime.now().strftime("_%d_%m_%H:%M") + '.png'
    plt.plot(range(len(running_rewards)), running_rewards, lw=2, color=np.random.rand(3, ), label=label_str)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Running average of Rewards')
    plt.legend()
    plt.savefig('logs/'+ file_str)
    plt.show()

    return ep_rewards, running_rewards


def perform_update(policy, learning_rate, gamma = 0.9):
    '''

       :param policy: Policy object (neural network from pytorch) that will be trained
       :param learning_rate: learning rate for the optimizer
       :param gamma: Discount factor from the REINFORCE algorithm
    '''

    # Define the optimizer and set the learning rate
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Variable for the current return, loss
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

    # Normalizing the returns. Taken from a notebook from class
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # Product of the corresponding logprobs and returns.
    # This is used to compute the loss for backpropagation.
    for log_prob, G in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * G)

    # Reset the gradients of the parameters
    optimizer.zero_grad()

    # Compute the loss
    policy_loss = torch.cat(policy_loss).mean()

    # Backpropagate the loss
    policy_loss.backward()

    # Perform a parameter step in direction of the gradient
    optimizer.step()

    # Reset the saved rewards and log probabilities
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def train(policy, step_size, render, num_episodes, gamma, log_interval, random_seed):
    '''
    Does the necessary settings and starts the actual training afterwards.

    :param policy: the LFAPolicy to be trained
    :param step_size: the learning rate of the gradient descent steps
    :param render: render the environment or not
    :param num_episodes: the number of maximal episodes to be trained
    :param gamma: the discount factor of the REINFORCE Algo
    :param log_interval: in what interval shall update infos be printed on comand line
    :param random_seed: the random seed to be used during training
    :return:
    '''
    try:
        # set the random seed
        set_random_seed(random_seed)

        # set the file  name of the model
        file_str = 'models/NN' + datetime.now().strftime("2020_%d_%m_%H:%M") + 'params_' + str(step_size) + '_' + str(
            num_episodes) + '_' + str(gamma) + '_' + str(policy.random_seed)
        policy.file_name = file_str

        # start the training
        reinforce(policy, step_size, render, num_episodes, gamma, log_interval)

    # Save the model in case of Keyboard Interrupt
    except KeyboardInterrupt:
        policy.save(policy.file_name + '_interrupt')


def set_random_seed(random_seed):
    '''
    Sets the random seed defined in the LFAPolicy

    :param random_seed: the random seed used during training
    :return:
    '''
    if random_seed is None:
        random_seed = 123

    random.seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

