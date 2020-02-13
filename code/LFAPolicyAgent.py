import numpy as np

STDDEV = 1

class LinearFAPolicy:
    def __init__(self, poly_degree):

        # Degree of the polynomial feature
        self.poly_degree = poly_degree

        # Initialize the weight matrix
        # The number of states is fixed to 8 and the number of
        # actions is fixed to 2, according to
        # the lunar lander environment
        self.weights = np.zeros(((self.poly_degree+1)**8,2))

        self.saved_log_probs = []
        self.rewards = []

    def poly_features(self,state, n):
        '''
        Computes the polynomial feature vector from the input states

        :param: state: Input state vector of size k (1D numpy array)
        :param: n: polynomial degree
        :return: feature vector phi consisting of (n+1)^k elements (1D numpy array)
        '''

        k = state.shape[0]
        phi = np.zeros((n + 1) ** k)

        # calculate the c-vectors from Sutton and Burato p. 211
        c = np.zeros(((n + 1) ** k, k))
        num = 0
        for pos in range(k):
            for row in range((n + 1) ** k):
                c[row][pos] = num
                if (row + 1) % ((n + 1) ** pos) == 0:
                    if num < n:
                        num += 1
                    else:
                        num = 0

        # calculate the feature vector phi
        for i in range(len(phi)):
            phi[i] = 1
            for j in range(k):
                phi[i] *= (state[j] ** c[i][j])

        return phi

    def select_action(self, state):
        ''' 
        Selects an action from the given observed state, by using
        the policy.

        :param: state: Input state vector of size 8 (1D numpy array)
        :return: Continuous action vector of size 2 (1D numpy array)
        '''

        # Compute feature vector
        feature_vector = self.poly_features(state,self.poly_degree)
        
        # Multiply feature vector with weight vector
        output_units = feature_vector.T@self.weights

        # Select Action 0, by sampling from a gaussian
        mean_0 = output_units[0]
        h_0 = np.random.normal(loc=mean_0,scale=STDDEV)
        action_0 = np.tanh(h_0)

        # Select Action 1, by sampling from a gaussian
        mean_1 = output_units[1]
        h_1 = np.random.normal(loc=mean_1,scale=STDDEV)
        action_1 = np.tanh(h_1)

        return action_0, action_1


    def save(self, state_file='models/LFAPolicy.npy'):
        # Save the weight matrix to file 
        np.save(state_file,self.weights)

    def load(self,state_file='models/LFAPolicy.npy'):
        # Load the weight matrix from file
        self.weights = np.load(state_file)
