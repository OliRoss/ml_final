import LFAPolicyAgent
import train_policy
import argparse

parser = argparse.ArgumentParser('REINFORCE algo to learn lunar lander')
parser.add_argument('gamma', metavar='gamma', type=float, nargs=1,
                    help='discount factor gamma', action='store')
parser.add_argument('poly_degree', metavar='poly_degree', type=int, nargs=1,
                    help='which degree of poly_degree to be used', action='store')
parser.add_argument('render', metavar='render', type=str, nargs=1, help='wether or not to render the environment graphically',
                    action='store')
parser.add_argument('step_size', metavar='step_size', type=float, nargs=1, help='the step size to take during the stochastic gradient descent',
                    action='store')
parser.add_argument('num_episodes', metavar='num_episodes', type=int, nargs=1, help='the number of episodes to train for',
                    action='store')
parser.add_argument('log_interval', metavar='log_interval', type=int, nargs=1, help='how many episodes to wait between print logs',
                    action='store')
args = parser.parse_args()

gamma = args.gamma[0]
poly_degree = args.poly_degree[0]
render = args.render[0]
step_size = args.step_size[0]
num_episodes = args.num_episodes[0]
log_interval = args.log_interval[0]

print("Params: {} {} {} {} {}".format(gamma,poly_degree,render,step_size, num_episodes))

policy = LFAPolicyAgent.LFAPolicy(poly_degree)
train_policy.reinforce(policy, step_size, render,num_episodes, gamma, log_interval)
