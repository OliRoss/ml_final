import numpy as np
import gym
import random
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make('LunarLanderContinuous-v2')

# Define hyperparameters
RANDOM_SEED = 123

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
    running_reward = 1.0

    # Lists to store the episodic and running rewards for plotting
    ep_rewards = list()
    running_rewards = list()

    # Generate episodes:
    for i in range(num_episodes):

        # Reset reward and state:
        ep_reward, state = 0, env.reset()

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

        norm_0 = np.linalg.norm(policy.weights)
        weight_0 = policy.weights[50,0]
        weight_1 = policy.weights[50,1]

        if i % log_interval == 0:
            print("Finished episode {}\tLast reward {:.2f}\tAverage reward: {:.2f}".format(
                i, ep_reward, running_reward))

    policy.save()


    # Plot the running average results only for the different settings
    fig = plt.figure(0, figsize=(20, 8))
    plt.rcParams.update({'font.size': 18})

    hp = {'name': 'linearFA', 'gamma': 0.9, 'poly_degree': 1, 'learning_rate': 5e-2}
    label_str = hp['name'] + '($\gamma$:' + str(hp['gamma']) + ',poly:' + str(hp['poly_degree']) + ',lr:' + str(
        hp['learning_rate']) + ')'
    plt.plot(range(len(running_rewards)), running_rewards, lw=2, color=np.random.rand(3, ), label=label_str)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Running average of Rewards')
    plt.legend()
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

    # Multiply returns by Gamma to the power of T
    num_steps = len(returns)
    returns = [x * gamma**(num_steps) for x in returns]

    print(len(policy.saved_log_probs))
    print(len(policy.saved_log_probs[0]))

    # Update the weights of the policy
    for i in range(num_steps):
        #print('length of non zero 2-tuple {}'.format(np.sum(np.array([1 if x.any() != 0 else 0 for x in policy.saved_log_probs[i]]))))
        #print(len(policy.saved_log_probs[i]))
        #print(len(policy.saved_log_probs[i][0]))
        #print(policy.saved_log_probs[i])
        policy.weights = policy.weights + step_size * returns[i] * policy.saved_log_probs[i]