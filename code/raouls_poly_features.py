import numpy as np

class test():
    def __init__(self,poly_degree):
        self.poly_degree = poly_degree
        self.c_vector = test.calc_c(self.poly_degree)

    @staticmethod
    def calc_c(poly_degree):
        '''
        Computes the c vector from Sutton and Burato p. 211

        :param poly_degree: polynomial degree
        :return: vector with the components for the polynomial feature function
        '''
        c = np.zeros(((poly_degree + 1) ** 8, 8))
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

    def poly_features(self, state, n):
        '''
        Computes the polynomial feature vector from the input states

        :param: state: Input state vector of size k (1D numpy array)
        :param: n: polynomial degree
        :return: feature vector phi consisting of (n+1)^k elements (1D numpy array)
        '''

        phi = np.zeros((n + 1) ** 8)

        # calculate the feature vector phi
        for i in range(len(phi)):
            phi[i] = 1
            for j in range(8):
                phi[i] *= (state[j] ** self.c_vector[i][j])

        return phi
