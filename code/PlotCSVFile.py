import matplotlib.pyplot as plt
import numpy as np
import os

import argparse

# Parse command line options
parser = argparse.ArgumentParser('python3 PlotCSVFile.py')
parser.add_argument('file_1', metavar='csv_file_1', type=str, nargs=1,
                     help='first file in csv format', action='store')
parser.add_argument('file_2', metavar='csv_file_2', type=str, nargs=1,
                     help='second file in csv format', action='store')
parser.add_argument('-o', action='store', dest='out_file',metavar="output_file", help='optional output file name. '
                                                'If none is given, the plot is only shown and not saved.')
parser.add_argument('-l1', action='store', dest='label_1', metavar="plot_label_1",help='optional '
                                                                                       'label for first dataset')
parser.add_argument('-l2', action='store', dest='label_2', metavar="plot_label_2",help='optional '
                                                                                       'label for second dataset')

args = parser.parse_args()

# Set the name of the output file and the graph labels
out_file = args.out_file
label_1 = args.label_1
label_2 = args.label_2

# Read the name of the input files
file_1 = os.path.splitext(os.path.basename(args.file_1[0]))[0]
file_2 = os.path.splitext(os.path.basename(args.file_2[0]))[0]

# Read the rewards from the provided csv files
rewards_1 = np.loadtxt(args.file_1[0], delimiter=",")
rewards_2 = np.loadtxt(args.file_2[0], delimiter=",")

# Set labels, if none are given
if label_1 is None:
    label_1 = file_1
if label_2 is None:
    label_2 = file_2

# Plot the rewards
fig = plt.figure(0, figsize=(20, 8))
plt.rcParams.update({'font.size': 18})
plt.plot(range(len(rewards_1)), rewards_1, lw=2, color=np.random.rand(3, ), label=label_1)
plt.plot(range(len(rewards_2)), rewards_2, lw=2, color=np.random.rand(3, ), label=label_2)
plt.grid()
plt.xlabel('Episode #')
plt.ylabel('Reward')
plt.legend()

# Save plot to file, if the corresponding flag is used
if out_file is not None:
    plt.savefig(out_file)
plt.show()
