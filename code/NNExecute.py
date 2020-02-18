import NNPolicyAgent
import NNTrainPolicy
import argparse

# Parse command line options
parser = argparse.ArgumentParser('REINFORCE algo to learn lunar lander')
parser.add_argument('gamma', metavar='gamma', type=float, nargs=1,
                    help='discount factor gamma', action='store')
parser.add_argument('render', metavar='render', type=str, nargs=1, help='wether or not to render the environment '
                                                                        'graphically',
                    action='store')
parser.add_argument('learning_rate', metavar='learning_rate', type=float, nargs=1, help='the learning rate to take '
                                                                        'during the stochastic gradient descent',
                    action='store')
parser.add_argument('num_episodes', metavar='num_episodes', type=int, nargs=1, help='the number of episodes '
                                                                                    'to train for',
                    action='store')
parser.add_argument('log_interval', metavar='log_interval', type=int, nargs=1, help='how many episodes to wait between'
                                                                                    ' print logs',
                    action='store')
parser.add_argument('-r', action='store', dest='random_seed', type=int, help='sets the random seed')
args = parser.parse_args()

gamma = args.gamma[0]
render = args.render[0]
learning_rate = args.learning_rate[0]
num_episodes = args.num_episodes[0]
log_interval = args.log_interval[0]
random_seed = args.random_seed

# Create neural network policy agent
policy = NNPolicyAgent.NNPolicy(random_seed)

# Train the neural network policy agent
NNTrainPolicy.train(policy, learning_rate, render, num_episodes, gamma, log_interval, random_seed)
