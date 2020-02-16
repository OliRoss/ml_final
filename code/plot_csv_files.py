import matplotlib.pyplot as plt
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser('Plots two csv files in the same plot and uses their file names as labels')
parser.add_argument('file_1', metavar='file_1', type=str, nargs=1,
                     help='First file in csv format', action='store')
parser.add_argument('file_2', metavar='file_2', type=str, nargs=1,
                     help='Second file in csv format', action='store')

args = parser.parse_args()


file_1 = os.path.splitext(os.path.basename(args.file_1[0]))[0]
file_2 = os.path.splitext(os.path.basename(args.file_2[0]))[0]

rewards_1 = np.loadtxt(args.file_1[0], delimiter=",")
rewards_2 = np.loadtxt(args.file_2[0], delimiter=",")

label_1 = file_1
label_2 = file_2

# Plot the rewards
fig = plt.figure(0, figsize=(20, 8))
plt.rcParams.update({'font.size': 18})

# Plot
plt.plot(range(len(rewards_1)), rewards_1, lw=2, color=np.random.rand(3, ), label=label_1)
plt.plot(range(len(rewards_2)), rewards_2, lw=2, color=np.random.rand(3, ), label=label_2)

plt.grid()
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend()
plt.savefig("comparison.png")
plt.show()