import numpy as np
import LFAPolicyAgent
import NNPolicyAgent
import RandomAgent
import gym
from _datetime import datetime
import argparse
import os

# Parse command line options
parser = argparse.ArgumentParser('python3 EvalPolicy.py')
parser.add_argument('num_episodes', metavar='num_episodes', type=int, nargs=1,
                    help='amount of episodes that the agent will be evaluated for', action='store')
parser.add_argument('-f', action='store', type=str, dest='file', help='optional filename for saved models.'
                                                         ' If none is given, a random agent is evaluated.')
parser.add_argument('-o', action='store', dest='out_file', help='optional filename for csv-Data')
parser.add_argument('-r', action='store', type=int, dest='random_seed', help='optional random seed for the environment.'
                                                         ' If none is given, 123 will be used.')
parser.add_argument('-p', action='store', type=int, dest='poly_degree', help='polynomial degree of the LFA Agent.'
                                                         ' Only needed for LFA agents')
args = parser.parse_args()

num_episodes = args.num_episodes[0]
poly_degree = args.poly_degree
out_file = args.out_file
random_seed = args.random_seed
file = args.file

# Set random seed if none is given
if random_seed is None:
    random_seed = 123

# Initialize policy objects and load weights
if args.file is not None:
    file = os.path.splitext(os.path.basename(args.file))

    # Neural network policy
    if file[1] == '.pt':
        policy = NNPolicyAgent.NNPolicy(random_seed)
        policy.load(args.file)
        file_str = 'logs/' + 'neural_net' + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'

    # LFA policy
    elif file[1] == '.npy':
        if poly_degree is None:
            raise Exception("Please provide the polynomial degree of the LFA Agent.")
        policy = LFAPolicyAgent.LFAPolicy(poly_degree,random_seed)
        policy.load(args.file)
        file_str = 'logs/' + 'lfa_agent_degree_' + "{}".format(poly_degree) + \
                   datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'
    else:
        raise Exception('Please provide valid input. See EvalPolicy.py -h for help.')
else:
    # Random Agent
    policy = RandomAgent.random_policy()
    file_str = 'logs/' + 'random_agent' + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'

# Define file name of output file if none is given
if out_file is not None:
    file_str = out_file

# Initialize environment and seeds
env = gym.make('LunarLanderContinuous-v2')
env.seed(random_seed)
np.random.seed(random_seed)

# Perform evaluation, using the policies' built-in method
rewards = policy.evaluate(num_episodes)

# Save to file
np.savetxt(file_str,rewards, delimiter=",")
