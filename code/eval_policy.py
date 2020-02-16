import numpy as np
import LFAPolicyAgent
import NNPolicyAgent
import random_agent
import gym
import matplotlib.pyplot as plt

NUM_EPISODES = 10
env = gym.make('LunarLanderContinuous-v2')

# Create Agents
random_policy = random_agent.random_policy()
# lfa_policy_1 = LFAPolicyAgent.LFAPolicy(1)
# lfa_policy_2 = LFAPolicyAgent.LFAPolicy(2)
# lfa_policy_3 = LFAPolicyAgent.LFAPolicy(3)
NN_policy = NNPolicyAgent.NNPolicy()

# Load saved weight vector
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

# Plot the rewards
fig = plt.figure(0, figsize=(20, 8))
plt.rcParams.update({'font.size': 18})

label_str_random = 'random_policy'
# label_str_1 = 'trained_policy, poly_degree = 1'
# label_str_2 = 'trained_policy, poly_degree = 2'
# label_str_3 = 'trained_policy, poly_degree = 3'
label_str_nn = 'trained_policy, NN'

plt.plot(range(len(random_rewards)), random_rewards, lw=2, color=np.random.rand(3, ), label=label_str_random)
# plt.plot(range(len(lfa_1_rewards)), lfa_1_rewards, lw=2, color=np.random.rand(3, ), label=label_str_1)
# plt.plot(range(len(lfa_2_rewards)), lfa_2_rewards, lw=2, color=np.random.rand(3, ), label=label_str_2)
# plt.plot(range(len(lfa_3_rewards)), lfa_3_rewards, lw=2, color=np.random.rand(3, ), label=label_str_3)
plt.plot(range(len(nn_rewards)), nn_rewards, lw=2, color=np.random.rand(3, ), label=label_str_nn)

plt.grid()
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend()
plt.show()
