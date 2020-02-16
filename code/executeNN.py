import NNPolicyAgent
import train_policy_NN
import argparse

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
parser.add_argument('-f', action="store", dest="file", help="loads a pre trained model if given and trains that model")
args = parser.parse_args()

gamma = args.gamma[0]
render = args.render[0]
learning_rate = args.learning_rate[0]
num_episodes = args.num_episodes[0]
log_interval = args.log_interval[0]
file = args.file

print("Params: {} {} {} {} {}".format(gamma,render,learning_rate, num_episodes, file))

policy = NNPolicyAgent.NNPolicy()

if file is not None:
    policy.load(file)
    if len(policy.weights) != (policy.poly_degree + 1)**8:
        raise ValueError('The given polynomial degree is not compatible with the loaded weight matrix')
train_policy_NN.reinforce(policy, learning_rate, render, num_episodes, gamma, log_interval)
