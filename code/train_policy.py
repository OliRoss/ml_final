import numpy as np
import gym
import random

# Initialize the environment
env = gym.make('LunarLanderContinuous-v2')

# Define hyperparameters
RANDOM_SEED = 123
LOG_INTERVAL = 1
NUM_EPISODES = 100
GAMMA = 0.9

# Set seeds for reproducability
random.seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def reinforce(policy, step_size, render=False):
    '''
    Implements the REINFORCE algorithm.
    
    :param: policy: LFAPolicy object to be trained
    :param: step_size: step size of the gradient descent
    :param: render: wether or not to render the environment
    '''

    # Generate episodes:
    for i in range(NUM_EPISODES):

        # Reset reward and state:
        reward, state = 0, env.reset()

        # For each step
        for t in range (1, 10000):

            # Select action according to policy:
            action_0, action_1 = policy.select_action(state)

            # Perform action and observe state + reward
            state, reward, done, _ = env.step(np.array([action_0,action_1]))

            if render:
                env.render()

            # Store the reward
            policy.rewards.append(reward)

            # stop episode if done
            if done:
                break

        # Perform the gradient update for the current episode
        perform_update(policy,step_size)
        
        if i % LOG_INTERVAL == 0:
            print("Finished episode {}".format(i))
        

def perform_update(policy, step_size):
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
        G = r + GAMMA * G
        returns.insert(0, G)

    # Multiply returns by Gamma to the power of T
    num_steps = len(returns)
    returns = [x * GAMMA**(num_steps) for x in returns]

    # Update the weights of the policy 
    for i in range(num_steps):
        policy.weights = policy.weights + step_size * returns[i] * policy.saved_log_probs[i] 
