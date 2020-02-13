from train_policy import train_policy

# A list for storing the hyperparameters and the corresponding results
results = list()

# Run the REINFORCE algorithm for 3 settings

# Setting 1
hyperparam_dict = {'name': 'linearFA', 'gamma':0.9, 'poly_degree':5, 'learning_rate':5e-3}
ep_rewards, running_rewards = train_policy(**hyperparam_dict)
# Store the results
results.append((hyperparam_dict, ep_rewards, running_rewards))
