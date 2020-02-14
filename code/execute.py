import LFAPolicyAgent
import train_policy

policy = LFAPolicyAgent.LFAPolicy(1)
train_policy.reinforce(policy, 0.05, False)

