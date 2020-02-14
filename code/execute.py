import LFAPolicyAgent
import train_policy

if __name__ == '__main___':
    policy = LFAPolicyAgent.LFAPolicy(1)
    train_policy.reinforce(policy, 0.05, True)

