import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Initialize the environment
env = gym.make('LunarLanderContinuous-v2')

# Define hyperparameters
RANDOM_SEED = 123
SAVE_INTERVAL = 1

# Set seeds for reproducability
random.seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def reinforce(policy, step_size, render=False, num_episodes=100, gamma=0.9,log_interval=1):
    '''
    Implements the REINFORCE algorithm.
    
    :param: policy: LFAPolicy object to be trained
    :param: step_size: step size of the gradient descent
    :param: render: wether or not to render the environment
    '''
    print("render: {}".format(render))
    # To track the reward across consecutive episodes (smoothed)
    running_reward = -250
    best_running_reward = -250

    # Lists to store the episodic and running rewards for plotting
    ep_rewards = list()
    running_rewards = list()

    # Generate episodes:
    for i in range(num_episodes):

        # Reset reward and state:
        ep_reward, state = 0, env.reset()

        start = time.time()

        # For each step
        for t in range (1, 10000):

            # Select action according to policy:
            action_0, action_1 = policy.select_action(state)

            # Perform action and observe state + reward
            state, reward, done, _ = env.step(np.array([action_0,action_1]))

            if render == "True":
                env.render()

            # Store the reward
            policy.rewards.append(reward)

            # Track the total reward in this episode
            ep_reward += reward

            # stop episode if done
            if done:
                break

        # Update the running reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Store the rewards for plotting
        ep_rewards.append(ep_reward)
        running_rewards.append(running_reward)

        # Perform the gradient update for the current episode
        perform_update(policy,step_size,gamma)

        norm = np.linalg.norm(policy.weights)
        weight_0 = policy.weights[50,0]
        weight_1 = policy.weights[50,1]

        end = time.time()

        if i % log_interval == 0:
            print("Finished episode {} in {:.2f} seconds\tSteps: {}\tLast reward {:.2f}\tAverage reward: {:.2f}\t\tNorm: {:.5f}\tWeights: {:.5f} {:.5f}".format(
                i, end - start, t, ep_reward, running_reward, norm, weight_0, weight_1))
        # Stopping criteria
        if running_reward > best_running_reward:
            policy.save()
            best_running_reward = running_reward
        if running_reward > env.spec.reward_threshold:
            print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
            break
        # Stopping criteria
        if norm > 100000000000:
            print('Norm explodes. Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
            break

    policy.save()

    # Plot the running average results
    fig = plt.figure(0, figsize=(20, 8))
    plt.rcParams.update({'font.size': 18})

    hp = {'name': 'linearFA', 'gamma': gamma, 'poly_degree': policy.poly_degree, 'learning_rate': step_size}
    label_str = hp['name'] + '(gamma:' + str(hp['gamma']) + ',poly:' + str(hp['poly_degree']) + ',lr:' + str(
        hp['learning_rate']) + ')'
    file_str = label_str + datetime.now().strftime("_%d_%m_%H:%M") + '.png'
    plt.plot(range(len(running_rewards)), running_rewards, lw=2, color=np.random.rand(3, ), label=label_str)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Running average of Rewards')
    plt.legend()
    plt.savefig('logs/' + file_str)
    plt.show()
        

def perform_update(policy, step_size, gamma=0.9):
    '''
    Performs the gradient ascent update for one episode in the REINFORCE algorithm.

    :param: policy: LFAPolicy object to be trained
    :param: step_size: step size of the gradient descent
    '''

    # Variable for the current return
    G = 0
    returns = []

    # Go through the list of observed rewards and calculate the returns
    for r in policy.rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)

    average_reward = np.mean(policy.rewards)

    # Define a small float which is used to avoid divison by zero
    eps = np.finfo(np.float32).eps.item()
    # Normalize returns by subtracting the mean and dividing by the standard deviation
    arr = np.array(returns)
    mean = arr.mean()
    std = arr.std()
    returns = [(x - mean) / (std + eps) for x in returns]

    # Multiply returns by Gamma to the power of T
    num_steps = len(returns)
    returns = [x * gamma**(num_steps) for x in returns]

    # Update the weights of the policy
    for i in range(num_steps):

        # Gradient step without baseline
        policy.weights = policy.weights + step_size * returns[i] * policy.saved_log_probs[i]

        # Gradient step with baseline
        # policy.weights = policy.weights + step_size * (returns[i]-average_reward) * policy.saved_log_probs[i]

    # delete the probabilities and rewards
    del policy.saved_log_probs[:]
    del policy.rewards[:]

def train(policy, step_size, render,num_episodes, gamma, log_interval):
    file_str = 'models/' + datetime.now().strftime("2020_%d_%m_%H:%M") + 'params_' + str(step_size) + '_' + str(num_episodes) + '_' + str(gamma) + '_' + str(policy.poly_degree)
    policy.file_name = file_str
    reinforce(policy, step_size, render, num_episodes, gamma, log_interval)