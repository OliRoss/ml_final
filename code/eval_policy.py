import numpy as np
import LFAPolicyAgent
import NNPolicyAgent
import random_agent
import gym
import matplotlib.pyplot as plt
from _datetime import datetime
import argparse
import os


parser = argparse.ArgumentParser('Evaluate an agent on the lunar lander environment.')
parser.add_argument('num_episodes', metavar='num_episodes', type=int, nargs=1,
                    help='Amount of episodes that the agent will be evaluated for', action='store')
parser.add_argument('-f', action='store', dest='file', help='Optional filename for saved models. If none is given, '
                                                            'a random agent is evaluated.')
parser.add_argument('-o', action='store', dest='out_file', help='Optional filename for csv-Data')
parser.add_argument('-r', action='store', type=int, dest='random_seed', help='Optional random seed for the environment.'
                                                                             'If none is given, 123 will be used.')
parser.add_argument('-p', action='store', type=int, dest='poly_degree', help='Polynomial degree of the LFA Agent.'
                                                                             'Only needed for LFA agents')
args = parser.parse_args()

num_episodes = args.num_episodes[0]
poly_degree = args.poly_degree
out_file = args.out_file
random_seed = args.random_seed

if random_seed is None:
    random_seed = 123

print("Params: \n file: {}\n num_episodes: {}\n poly_degree: {}\n "
      "out_file: {}\n random_seed: {}\n".format(file,num_episodes,poly_degree,out_file,random_seed))

# Initialize objects and load weights
if args.file is not None:
    file = os.path.splitext(os.path.basename(args.file))
    if file[1] == '.pt':
        if poly_degree is None:
            raise Exception("Please provide the polynomial degree of the LFA Agent.")
        policy = NNPolicyAgent.NNPolicy(random_seed)
        policy.load(args.file[0])
        file_str = 'logs/' + 'neural_net' + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'
    elif file[1] == '.npy':
        policy = LFAPolicyAgent.LFAPolicy(poly_degree,random_seed)
        policy.load(args.file[0])
        file_str = 'logs/' + 'lfa_agent_degree_' + "{}".format(poly_degree) + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'
    else:
        raise Exception('Please provide valid input. See eval_policy.py -h for help.')
else:
    policy = random_agent.random_policy()
    file_str = 'logs/' + 'random_agent' + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'

if out_file is not None:
    file_str = out_file

# Initialize environment
env = gym.make('LunarLanderContinuous-v2')
env.seed(random_seed)
np.random.seed(random_seed)

# Perform evaluation
rewards = policy.evaluate(num_episodes)

# Save to file
np.savetxt(file_str,rewards, delimiter=",")
