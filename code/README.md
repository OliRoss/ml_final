Assignment 6
======================================

This is the code base for the final assignment done by:

Oliver Ross (oliver.ross@student.uibk.ac.at),
Raoul Schikora (raoul.schikora@student.uibk.ac.at)

We compare two Agents for the Lunar Lander task. One agent uses linear function
approximation (LFAPolicyAgent) and the other one uses a neural network
(NNPolicyAgent). Both agents are trained by the REINFORCE algorithm.

Prerequisites
-------------------------------------

For matrix operations and PyTorch install NumPy:

```
pip3 install numpy
```

Furthermore, PyTorch is used for the NNPolicyAgent:

```
pip3 install torch
```

The code uses the LunarLanderContinuous-v2 environment provided by gym:

```
pip3 install gym[box2d]
```

Run the code
---------------------------------------

For usage information call the mentioned functions below with the -h flag.

To train a LFAPolicyAgent with discount factor 0.9, using a feature vector of
polynomial degree 3, without rendering the environment, a learning rate of
0.02 for 1000 episodes and a log interval of 10 run:

```
python3 LFAExecute.py 0.9 3 False 0.02 1000 10 -r 2611
```

The Flag -r sets the random seed.

To train a NNPolicyAgent with discount factor 0.9, without rendering the
environment, a learning rate of 0.05 for 5000 episodes and a log interval
of 10 run:

```
python3 NNExecute.py 0.9 False 0.05 5000 10
```

During execution the episodes are logged in a .csv-file. To plot and compare
two files in one figure call:

```
python3 PlotCSVFile.py <file1> <file2>
```

To deterministically evaluate a trained agent for 100 episodes 
call a .npy or a .pt model with:

```
python3 EvalPolicy.py 100 -f <model>
```

For a LFAPolicyAgent use the -p flag for the needed polynomial degree of
the feature vector.