import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist


class NNPolicy(nn.Module):
    def __init__(self, random_seed=123):
        super(NNPolicy, self).__init__()

        '''
        Neural network with the following architecture:
        8 unit input layer
        64 unit fully connected layer
        64 unit fully connected layer
        16 unit fully connected layer
        4 unit fully connected layer
        
        All layers have dropout layers with p = 0.6 in between. 
        All activation functions are ReLU, except for the output units.
        
        '''
        self.random_seed = random_seed
        self.affine1 = nn.Linear(8, 64)
        self.dropout1 = nn.Dropout(p=.6)
        self.affine2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=.6)
        self.affine3 = nn.Linear(64, 16)
        self.dropout3 = nn.Dropout(p=.6)
        self.affine4 = nn.Linear(16, 4)

        # Final activation is tanh to squeeze values into range [-1,1]
        self.activation = nn.Tanh()

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        '''
        Feed-forward function of the neural network

        :param x: Input to the neural network (8-dimensional state vector)
        :return: Output of the neural network
        '''

        x = F.relu(self.dropout1(self.affine1(x)))
        x = F.relu(self.dropout2(self.affine2(x)))
        x = F.relu(self.dropout3(self.affine3(x)))
        x = self.activation(self.affine4(x))

        return x

    def gaussian_policy(self, state):

        '''
        Implements a policy that samples actions from a gaussian distribution. The mean
        and standard deviation of the gaussian is computed by the neural network.

        :param state: Input to the neural network (8-dimensional state vector)
        :return: Mean of the computed gaussian, sampled actions and log_prob of the sample actions
        '''

        # Unsqueeze Pytorch tensor
        state = torch.from_numpy(state).float().unsqueeze(0)

        # Get the gaussian mean and standard deviation from the neural network
        output = self.forward(state)
        mu = output[:,0:2]
        std = output [:,2:4]

        # Set up gaussian, with computed mean and standard deviation
        dist = torch_dist.Normal(mu, std)

        # Sample from gaussian
        action = dist.sample()

        # Find log_prob of the action
        log_prob = dist.log_prob(action)

        # Save the computed log_prob
        self.saved_log_probs.append(log_prob)

        # Return mean, action and log_prob
        return output, action, log_prob

    def save(self, state_file='models/policy_network.pt'):
        '''
        Saves the neural network to disk

        :param state_file: File location
        :return:
        '''

        # Save the model state
        torch.save(self.state_dict(), state_file + '.pt')
        print('Policy saved at ' + state_file)

    @staticmethod
    def load(state_file='models/policy_network.pt'):
        '''
        Loads a neural network from disk

        :param state_file: File location
        :return:
        '''

        # Create a network object with the constructor parameters
        policy = NNPolicy()

        # Load the weights
        policy.load_state_dict(torch.load(state_file))
        print('Policy loaded from ' + state_file)

        # Set the network to evaluation mode
        policy.eval()
        return policy

    def evaluate(self,num_episodes):
        '''
        Function for evaluating the Policy using deterministic action selection used
        for comparison between the policy and an random agent.

        :param num_episodes: the number of episode to be evaluated
        :return: list of rewards per episode
        '''

        # Set up the environment
        env = gym.make('LunarLanderContinuous-v2')
        rewards = []

        # Counter for the number of episodes (for logging purposes)
        i = 0

        # Create episodes until the specified limit is reached
        for episode in range(num_episodes):

            # Print progress
            if i % 10 == 0:
                print("NN Policy: Evaluating episode #{}".format(i))
            i = i + 1

            # Reset environment and get first state
            observation = env.reset()
            episode_reward = 0

            # Iterate through the steps of the episode
            while True:
                # Unsqueeze Pytorch tensor
                state = torch.from_numpy(observation).float().unsqueeze(0)

                # Get outputs from neural network
                output = self.forward(state)

                # The first two elements are the action
                action = output[:, 0:2]

                # Observe the reaction of the environment
                observation, reward, done, info = env.step(np.array([action[0][0], action[0][1]]))

                # Compute episode reward
                episode_reward += reward
                if done:
                    rewards.append(episode_reward)
                    break

        return rewards
