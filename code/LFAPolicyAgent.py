import numpy as np
import gym

STDDEV = np.exp(-.5)


class LFAPolicy:
    def __init__(self, poly_degree, random_seed=123):

        # Initialize the random seed to be used during training
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        # Degree of the polynomial feature
        self.poly_degree = poly_degree

        # Initialize the weight matrix
            # The number of states is fixed to 8 and the number of actions is fixed to 2, according to
            # the lunar lander environment
        self.weights = np.zeros(((self.poly_degree+1)**8, 2))

        # Set up lists for the log probs and rewards
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

        # Initialize empty list
        c = [[0] * 8] * ((poly_degree + 1) ** 8)

        # Compute the c vector
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
        Computes the polynomial feature vector from the input state.

        :param: state: Input state vector of size 8 (1D list)
        '''

        feature_vector = [1] * ((self.poly_degree + 1) ** 8)

        # calculate the feature vector phi
        for i in range(len(feature_vector)):

            # Compute the specific feature, according to c vector coefficients.
                # (because of measured performance boost we do not use a for-loop)
            feature_vector[i] = state[0] ** self.c_vector[i][0] * state[1] ** self.c_vector[i][1] * \
                                state[2] ** self.c_vector[i][2] * state[3] ** self.c_vector[i][3] * \
                                state[4] ** self.c_vector[i][4] * state[5] ** self.c_vector[i][5] * \
                                state[6] ** self.c_vector[i][6] * state[7] ** self.c_vector[i][7]

        return feature_vector

    def select_action(self, state):
        ''' 
        Selects an action from the given observed state, by using
        the policy. It samples from a gaussian, where the mean is output by the
        linear function approximater.

        :param: state: Input state vector of size 8 (1D numpy array)
        :return: Continuous action vector of size 2 (1D numpy array)
        '''

        # Cast state to list for better performance
        state = state.tolist()

        # Compute feature vector
        feature_vector = np.array(self.poly_features(state))

        # Multiply feature vector with weight vector
        output_units = feature_vector.T @ self.weights

        # Compute the tanh to squeeze the output into the [-1,1] range
        mean = np.tanh(output_units)

        # Select actions, by sampling from a gaussian
        action = np.random.normal(loc=mean, scale=STDDEV)

        # Clip the selected action to the valid range
            # (will be done by the environment too, but we make it explicit for clarity)
        action = np.clip(action, -1, 1)

        # Compute log prob for given state and actions
        factor = (action - output_units)/(STDDEV**2)
        log_prob = np.array([factor[0] * feature_vector, factor[1] * feature_vector]).T

        # Save the computed log probs
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

        # Cast state to list for better performance
        state = state.tolist()

        # Compute feature vector
        feature_vector = np.array(self.poly_features(state))

        # Multiply feature vector with weight vector
        output_units = feature_vector.T @ self.weights

        # Return the computed actions
        return output_units

    def evaluate(self,num_episodes):
        '''
        Function for evaluating the Policy using deterministic action selection.
        Can be used for comparison between the policy and an random agent after training.

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
                print("LFA Policy: Evaluating episode #{}".format(i))
            i = i + 1

            # Reset environment and get first state
            observation = env.reset()
            episode_reward = 0

            # Iterate through the steps of the episode
            while True:

                # Select action deterministically
                action = self.select_action_deterministic(observation)

                # Observe the reaction of the environment
                observation, reward, done, info = env.step(action)

                # Compute episode reward
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

