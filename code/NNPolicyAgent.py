import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist

class NNPolicy(nn.Module):
    def __init__(self):
        super(NNPolicy, self).__init__()

        self.affine1 = nn.Linear(8, 100)
        self.affine2 = nn.Linear(100, 100)
        self.affine3 = nn.Linear(100, 2)

        self.activation = nn.Tanh()

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        '''
        Feed-forward function of the neural network

        :param x: Input to the neural network (8-dimensional state vector)
        :return: Output of the neural network
        '''

        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.activation(self.affine3(x))
        return x

    def gaussian_policy(self, x):

        '''
        Implements a policy that samples actions from a gaussian distribution. The mean
        of the gaussian is computed by the neural network.

        :param x: Input to the neural network (8-dimensional state vector)
        :return: Mean of the computed gaussian, sampled actions and log_prob of the sample actions
        '''
        # Unsqueeze Pytorch tensor
        x = torch.from_numpy(x).float().unsqueeze(0)

        # get the gaussian mean from the neural network
        mu = self.forward(x)

        # set constant log_std
        log_std = -0.5 * torch.ones(2)

        # Compute standard deviation
        std = torch.exp(log_std)

        # Set up gaussian, with given mean and standard deviation
        dist = torch_dist.Normal(mu, std)

        # Sample from gaussian
        action = dist.sample()

        # Find log_prob of the action
        log_prob = dist.log_prob(action)

        # Save the computed log_prob
        self.saved_log_probs.append(log_prob)

        # Return mean, action and log_prob
        return mu, action, log_prob

    def save(self, state_file='models/policy_network.pt'):
        '''
        Saves the neural network to disk

        :param state_file: File location
        :return:
        '''

        # Save the model state
        torch.save(self.state_dict(), state_file)
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

