import LFAPolicyAgent
import train_policy

policy = LFAPolicyAgent.LFAPolicy(2)
train_policy.reinforce(policy, 0.05, True)

