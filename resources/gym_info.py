import gym
# Initialize the environment
# Check the env source at
env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

# Information about observations
# Observation is an array of 8 numbers
# These numbers are usually in the range of -1 .. +1, but spikes can be higher
print('Shape of observations: {}'.format(env.observation_space.shape))
print('A few sample observations:')
for i in range(5):
    print(env.observation_space.sample())

# Information about actions
# Action is two floats [main engine, left-right engines].
# Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with
# less than 50% power.
# Left-right: -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

print(env.action_space.shape)
print('A few sample actions:')
for i in range(5):
    print(env.action_space.sample())
