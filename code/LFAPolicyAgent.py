import numpy as np
import gym
import time

STDDEV = np.exp(-.5)
#STDDEV = 0.3
LEARNING_RATE = 5e-2


class LFAPolicy:
    def __init__(self, poly_degree, random_seed=123):

        # initialize the random seed to be used during training
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        # Degree of the polynomial feature
        self.poly_degree = poly_degree

        # Initialize the weight matrix
        # The number of states is fixed to 8 and the number of
        # actions is fixed to 2, according to
        # the lunar lander environment
        self.weights = np.zeros(((self.poly_degree+1)**8, 2))

        self.saved_log_probs = []
        self.rewards = []

        # Store an instance of the c vector from Sutton and Burato p. 211.
        self.c_vector = LFAPolicy.calc_c(poly_degree)

        # file name of the weight matrix
        self.file_name = ''

    @staticmethod
    def calc_c(poly_degree):
        '''
        Computes the c vector from Sutton and Burato p. 211

        :param poly_degree: polynomial degree
        :return: vector with the components for the polynomial feature function
        '''
        c = [[0] * 8] * ((poly_degree + 1) ** 8)

        num = 0
        for pos in range(8):
            for row in range((poly_degree + 1) ** 8):
                c[row][pos] = num
                if (row + 1) % ((poly_degree + 1) ** pos) == 0:
                    if num < poly_degree:
                        num += 1
                    else:
                       num = 0

        return c

    def poly_features(self, state):
        '''
        Computes the polynomial feature vector from the input states

        :param: state: Input state vector of size k (1D list)
        '''

        feature_vector = [1] * ((self.poly_degree + 1) ** 8)

        # calculate the feature vector phi
        for i in range(len(feature_vector)):
            # beacause of performance we do not use a for-loop
            feature_vector[i] = state[0] ** self.c_vector[i][0] * state[1] ** self.c_vector[i][1] * \
                                state[2] ** self.c_vector[i][2] * state[3] ** self.c_vector[i][3] * \
                                state[4] ** self.c_vector[i][4] * state[5] ** self.c_vector[i][5] * \
                                state[6] ** self.c_vector[i][6] * state[7] ** self.c_vector[i][7]

        return feature_vector

    def select_action(self, state):
        ''' 
        Selects an action from the given observed state, by using
        the policy. It samples from a gaussian that is output by the
        linear function approximater.

        :param: state: Input state vector of size 8 (1D numpy array)
        :return: Continuous action vector of size 2 (1D numpy array)
        '''

        # cast state to list for performance reason
        state = state.tolist()

        # Compute feature vector
        feature_vector = np.array(self.poly_features(state))

        # Multiply feature vector with weight vector
        output_units = feature_vector.T @ self.weights

        mean = np.tanh(output_units)

        # Select Action 0, by sampling from a gaussian
        action = np.random.normal(loc=mean, scale=STDDEV)
        action = np.clip(action, -1, 1)

        # Compute log prob for given state and actions
        factor = (action - output_units)/(STDDEV**2)

        log_prob = np.array([factor[0] * feature_vector, factor[1] * feature_vector]).T
        self.saved_log_probs.append(log_prob)

        return action

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
                # env.render()
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
        np.save(file_name + '.npy', self.weights)
        print('Policy saved at ' + file_name + '.npy')

    def load(self, state_file='models/LFAPolicy.npy'):
        '''
        Function for loading a weight matrix

        :param state_file: file name
        '''
        # Load the weight matrix from file
        self.weights = np.load(state_file)
        print('Policy loaded from ' + state_file)

