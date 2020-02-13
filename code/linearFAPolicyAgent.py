import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class LinearFAPolicy(nn.Module):
    def __init__(self, feature_size, num_actions, feature_func, poly_degree):
        super(LinearFAPolicy, self).__init__()
        # Set the function that computes features
        self.feature_func = feature_func
        # Degree of the polynomial feature
        self.poly_degree = poly_degree

        self.affine1 = nn.Linear(feature_size, num_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.affine1(x)
        return F.tanh(action_scores)

    def select_action(self, state):
        # Convert the state to feature
        feature = self.feature_func(state, self.poly_degree)

        # Convert the feature from a numpy array to a torch tensor
        feature = torch.from_numpy(feature).float().unsqueeze(0)

        # Get the predicted probabilities from the policy network
        probs = self.forward(feature)

        # Sample the actions according to their respective probabilities
        m = Categorical(probs)
        action = m.sample()

        # Also calculate the log of the probability for the selected action
        self.saved_log_probs.append(m.log_prob(action))

        # Return the chosen action
        return action.item()

    def save(self, state_file='models/LinearFAPolicy.pt'):
        # Save the model state
        torch.save(self.state_dict(), state_file)

    @staticmethod
    def load(state_file='models/LinearFAPolicy.pt'):
        # Create a network object with the constructor parameters
        policy = nn.Module.Policy()
        # Load the weights
        policy.load_state_dict(torch.load(state_file))
        # Set the network to evaluation mode
        policy.eval()
        return policy
