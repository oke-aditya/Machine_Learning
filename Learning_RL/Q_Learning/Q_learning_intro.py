# Q-Learning is a model-free form of machine learning,
# in the sense that the AI "agent" does not need to know or have a model of the environment that it will be in. 
# The same algorithm can be used across a variety of environments.
# For a given environment, everything is broken down into "states" and "actions." The states are observations 
# and samplings that we pull from the environment, and the actions are the choices the agent has made based on the observation.
# there are "3" actions we can pass. This means, when we step the environment, we can pass a 0, 1, or 2 as our "action" for 
# each step. Each time we do this, the environment will 
# return to us the new state, a reward, whether or not the environment is done/complete, 
# and then any extra info that some envs might have.

# It doesnt matter to our model, but, for your understanding, 
# a 0 means push left, 1 is stay still, and 2 means push right. 
# We wont tell our model any of this, and that's the power of Q learning. 
# All the model needs to know is what the options for actions are,
# and what the reward of performing a chain of those actions would be given a state.

# How will Q-learning do that? So we know we can take 3 actions at any given time. That's our "action space."
#  Now, we need our "observation space." 
#  In the case of this gym environment, the observations are returned from resets and steps. 

import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

# What's a Q Table!?

# The way Q-Learning works is there's a "Q" value per action possible per state. 
# This creates a table. In order to figure out all of the possible states, we can either query the environment 
# (if it is kind enough to us to tell us)...or we just 
# simply have to engage in the environment for a while to figure it out.
# For the value at index 0,
#  we can see the high value is 0.6, the low is -1.2, and then for the value at index 1, the high is 0.07, and the low is -0.07. 

# We need to discretize these values. We use 20 buckets to do so.


# discrete_obs_size = [20] * len(env.observation_space.high)
discrete_obs_size = [20, 20]
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / discrete_obs_size

# print(discrete_obs_size)
# print(discrete_obs_win_size)   # Discrete size of our qtable


q_table = np.random.uniform(low=-2, high=0, size=(discrete_obs_size + [env.action_space.n]))
# print(q_table.shape)

# So, this is a 20x20x3 shape, which has initialized random Q values for us.
# The 20 x 20 bit is every combination of the bucket slices of all possible states. 
# The x3 bit is for every possible action we could take.

# So these values are random, and the choice to be between -2 and 0 is also a variable. 
# Each step is a -1 reward, and the flag is a 0 reward, 
# so it seems to make sense to make the starting point of random Q values all negative.

# We will consult with this table to determine our moves. 
# That final x3 is our 3 actions and each of those 3 actions have the "Q value" associated with them. 
# When we're being "greedy" and trying to "exploit" our environment, we will choose to go with the action that has the
# highest Q value for this state. Sometimes, however, especially initially, 
# we may instead wish to "explore" and just choose a random action. 
# These random actions are how our model will learn better moves over time.

# The DISCOUNT is a measure of how much we want to care about FUTURE reward 
# rather than immediate reward. Typically, this value will be fairly high, 
# and is between 0 and 1. We want it high because the purpose of Q Learning is indeed to learn a chain of events that 
# ends with a positive outcome, so it's only natural that we put greater importance 
# on long terms gains rather than short term ones.

# The max_future_q is grabbed after we've performed our action already, 
# and then we update our previous values based partially on the 
# next-step's best Q value. Over time, 
# once we've reached the objective once, this "reward" value gets slowly 
# back-propagated, one step at a time, per episode. 

# The LEARNING_RATE is between 0 and 1, same for discount.
# The EPISODES is how many iterations of the game we'd like to run.

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000      # Just to check
epsilon = 0.5          # Degree of random exploration
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Next, we need a quick helper-function that will convert our environment "state," to a "discrete" state instead:

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_obs_win_size
	return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
	if(episode % SHOW_EVERY == 0):
		render = True
		print("episode: %d" %(episode))
	else:
		render = False
	# Do for these many episodes
	discrete_state = get_discrete_state(env.reset())
	# print(discrete_state)
	# print(q_table[discrete_state])

	done = False

	while not done:
		# action = 2
		if(np.random.random() > epsilon):
			# Get action from the Q-Table
			action = np.argmax(q_table[discrete_state])
		else:
			# Perform a random action
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		new_discrete_state = get_discrete_state(new_state)
		
		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			cuurent_q = q_table[discrete_state + (action, )]
			# formula for new_q value that is used
			new_q = (1 - LEARNING_RATE) * cuurent_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)  
			# Update the q_table after the action is performed and reward is obtained.
			q_table[discrete_state + (action, )] = new_q

		elif new_state[0] >= env.goal_position:
			q_table[discrete_state + (action, )] = 0
			print("Yep!! We are done at episode no %d" %(episode))
		
		discrete_state = new_discrete_state	

	if(END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING):
		epsilon -= epsilon_decay_value
		# env.render()

env.close()