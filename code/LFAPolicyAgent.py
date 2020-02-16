import numpy as np
import gym
import random

STDDEV = np.exp(-.5)
#STDDEV = 0.3
LEARNING_RATE = 5e-2


class LFAPolicy:
    def __init__(self, poly_degree, random_seed):

        # Degree of the polynomial feature
        self.poly_degree = poly_degree

        # Initialize the weight matrix
        # The number of states is fixed to 8 and the number of
        # actions is fixed to 2, according to
        # the lunar lander environment
        self.weights = np.zeros(((self.poly_degree+1)**8, 2))

        self.saved_log_probs = []
        self.rewards = []

        # file name of the weight matrix
        self.file_name = ''

        # initialize the random seed to be used during training
        if random_seed is not None:
            self.random_seed = random_seed
        else:
            self.random_seed = 123


    def poly_features(self, state):
        '''
        Computes the polynomial feature vector from the input states

        :param: state: Input state vector of size k (1D numpy array)
        :param: n: polynomial degree
        :return: feature vector phi consisting of (n+1)^k elements (1D numpy array)
        '''

        phi = np.zeros((self.poly_degree + 1) ** 8)

        c = np.arange(0, self.poly_degree + 1)
        i = 0

        for p0 in c:
            for p1 in c:
                for p2 in c:
                    for p3 in c:
                        for p4 in c:
                            for p5 in c:
                                for p6 in c:
                                    for p7 in c:
                                        phi[i] = (state[0] ** p0) * (state[1] ** p1) * (state[2] ** p2) * (
                                                state[3] ** p3) * (state[4] ** p4) * (state[5] ** p5) * (
                                                         state[6] ** p6) * (state[7] ** p7)
                                        i += 1

        return phi

    def select_action(self, state):
        ''' 
        Selects an action from the given observed state, by using
        the policy. It samples from a gaussian that is output by the
        linear function approximater.

        :param: state: Input state vector of size 8 (1D numpy array)
        :return: Continuous action vector of size 2 (1D numpy array)
        '''

        # Compute feature vector
        feature_vector = self.poly_features(state)

        # Multiply feature vector with weight vector
        output_units = feature_vector.T @ self.weights

        # Select Action 0, by sampling from a gaussian
        h_0 = output_units[0]
        mean_0 = np.tanh(h_0)
        action_0 = np.random.normal(loc=mean_0, scale=STDDEV)

        # Select Action 1, by sampling from a gaussian
        h_1 = output_units[1]
        mean_1 = np.tanh(h_1)
        action_1 = np.random.normal(loc=mean_1, scale=STDDEV)

        # Compute log prob for given state and actions
        factor = (np.array([action_0,action_1]) - output_units)/(STDDEV**2)
        self.saved_log_probs.append(np.array([factor[0]*feature_vector,factor[1]*feature_vector]).T)

        return action_0, action_1

    def select_action_deterministic(self,state):
        '''
        Selects an action from the given observed state, by using
        the policy. It doesn't sample, but instead always chooses
        the mean output by the LFA, making it deterministic. It only
        exploits the learned policy without any exploration.

        :param: state: Input state vector of size 8 (1D numpy array)
        :return: Continuous action vector of size 2 (1D numpy array)
        '''

        # Compute feature vector
        feature_vector = self.poly_features(state)

        # Multiply feature vector with weight vector
        output_units = feature_vector.T @ self.weights

        # Return the computed actions
        return output_units[0], output_units[1]

    def evaluate(self,num_episodes):
        '''
        Function for evaluating the Policy using deterministic action selection used
        for comparison between the policy and an random agent.

        :param num_episodes: the number of episode to be evaluated
        :return: list of rewards per episode
        '''


        env = gym.make('LunarLanderContinuous-v2')
        rewards = []
        i = 0

        for episode in range(num_episodes):
            if i % 10 == 0:
                print("LFA Policy: Evaluating episode #{}".format(i))
            i = i + 1
            observation = env.reset()
            episode_reward = 0
            while True:
                action = self.select_action_deterministic(observation)
                observation, reward, done, info = env.step(action)
                env.render()
                episode_reward += reward
                if done:
                    rewards.append(episode_reward)
                    break

        return rewards

    def save(self, file_name):
        '''
        Function for saving the weight matrix to .npy-file

        :param state_file: file name
        '''
        # Save the weight matrix to file 
        np.save(self.file_name, self.weights)
        print('Policy saved at ' + file_name)

    def load(self, state_file='models/LFAPolicy.npy'):
        '''
        Function for loading a weight matrix

        :param state_file: file name
        '''
        # Load the weight matrix from file
        self.weights = np.load(state_file)
        print('Policy loaded from ' + state_file)

