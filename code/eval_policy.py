import numpy as np
import LFAPolicyAgent
import NNPolicyAgent
import random_agent
import gym
import matplotlib.pyplot as plt
from _datetime import datetime

NUM_EPISODES = 200
RANDOM_SEED = 123
env = gym.make('LunarLanderContinuous-v2')

env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create Agents
random_policy = random_agent.random_policy()
# lfa_policy_1 = LFAPolicyAgent.LFAPolicy(1)
# lfa_policy_2 = LFAPolicyAgent.LFAPolicy(2)
# lfa_policy_3 = LFAPolicyAgent.LFAPolicy(3)
NN_policy = NNPolicyAgent.NNPolicy()

# Load models
# lfa_policy_1.load()
# lfa_policy_2.load("models/2020_02_15_params_09_002_2000_rnd_1234.npy")
# lfa_policy_3.load()
NN_policy.load()

# Perform evaluation
random_rewards = random_policy.evaluate(NUM_EPISODES)
# lfa_1_rewards = lfa_policy_1.evaluate(NUM_EPISODES)
# lfa_2_rewards = lfa_policy_2.evaluate(NUM_EPISODES)
# lfa_3_rewards = lfa_policy_3.evaluate(NUM_EPISODES)
nn_rewards = NN_policy.evaluate(NUM_EPISODES)

# Define Labels
label_str_random = 'random_policy'
# label_str_1 = 'trained_policy, poly_degree = 1'
# label_str_2 = 'trained_policy, poly_degree = 2'
# label_str_3 = 'trained_policy, poly_degree = 3'
label_str_nn = 'trained_policy, NN'

# Plot the rewards
fig = plt.figure(0, figsize=(20, 8))
plt.rcParams.update({'font.size': 18})

# Plot
plt.plot(range(len(random_rewards)), random_rewards, lw=2, color=np.random.rand(3, ), label=label_str_random)
# plt.plot(range(len(lfa_1_rewards)), lfa_1_rewards, lw=2, color=np.random.rand(3, ), label=label_str_1)
# plt.plot(range(len(lfa_2_rewards)), lfa_2_rewards, lw=2, color=np.random.rand(3, ), label=label_str_2)
# plt.plot(range(len(lfa_3_rewards)), lfa_3_rewards, lw=2, color=np.random.rand(3, ), label=label_str_3)
plt.plot(range(len(nn_rewards)), nn_rewards, lw=2, color=np.random.rand(3, ), label=label_str_nn)

plt.grid()
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend()
plt.savefig("logs/"+label_str_nn + "_vs_" + label_str_random)
plt.show()

# Define file names
file_str_random = 'logs/' + label_str_random + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'
# file_str_lfa_1 =  'logs/' +label_str_lfa_1 + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'
# file_str_lfa_2 =  'logs/' +label_str_lfa_2 + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'
# file_str_lfa_3 = 'logs/ +label_str_lfa_3 + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'
file_str_nn = 'logs/' + label_str_nn + datetime.now().strftime("_%d_%m_%H:%M") + '_rewards.csv'

# Save to file
np.savetxt(file_str_random,random_rewards, delimiter=",")
# np.savetxt(file_str_lfa_1,lfa_1_rewards, delimiter=",")
# np.savetxt(file_str_lfa_2,lfa_2_rewards, delimiter=",")
# np.savetxt(file_str_lfa_3,lfa_3_rewards, delimiter=",")
np.savetxt(file_str_nn,nn_rewards, delimiter=",")
