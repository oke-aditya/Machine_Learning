import numpy as np
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
import pickle
import time
from matplotlib import style

style.use('ggplot')

size = 20                          # size of our environment
HM_episodes = 25000					# Episodes for training
move_penalty = 1                    # Penalty for moving
enemy_penalty = 300					# Hitting the enemy
food_reward = 25					# reward to eat the food

epsilon = 0.9
eps_decay = 0.9998

show_every = 1
start_q_table = None              # or existing q_table file

learning_rate = 0.1				 # learning_rate paramter 
discount = 0.95                  # discount factor for future rewards

player_n = 1
food_n = 2
enemy_n = 3

d = {1: (255,0,0), 2: (0,255,0), 3: (0,0,255)}       # Colors dictonary

class Blob:
	def __init__(self):
		self.x = np.random.randint(0, size)
		self.y = np.random.randint(0, size)

	def __str__(self):
		return f"{self.x}, {self.y}"

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)

	def action(self, choice):
		if choice == 0:
			self.move(x=1, y=1)
		elif choice == 1:
			self.move(x=-1, y=-1)
		elif choice == 2:
			self.move(x=-1, y=1)
		if choice == 3:
			self.move(x=1, y=-1)

	def move(self, x=False, y=False):
		if not x:
			self.x += np.random.randint(-1, 2)
		else:
			self.x += x
		if not y:
			self.y += np.random.randint(-1, 2)
		else:
			self.y += y
		
		# Checking for the boundary of observation space

		if self.x < 0:
			self.x = 0

		elif self.x > size-1:
			self.x = size - 1


		if self.y < 0:
			self.y = 0
			
		elif self.y > size-1:
			self.y = size - 1

if start_q_table is None:
	# We need to create a q_table
	q_table = {}
	for x1 in range(-size+1, size):
		for y1 in range(-size+1, size):
			for x2 in range(-size+1, size):
				for y2 in range(-size+1, size):
					q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
	with open(start_q_table, "rb" ) as f:
		q_table = pickle.load(f)

episode_rewards =[]

for episode in range(HM_episodes):
	player = Blob()
	food = Blob()
	enemy = Blob()

	if episode % show_every == 0:
		print("Episode no %d, epsion = %0.2f " %(episode, epsilon))
		print("Show")

		show = True
	else:
		show = False

	episode_reward = 0
	for i in range(200):
		obs = (player - food, player - enemy)
		if np.random.random() > epsilon:
			action = np.argmax(q_table[obs])
		else:
			action = np.random.randint(0,4)

		player.action(action)


		##### Maybe later  #####
		# enemy.move()
		# food.move()
		######################

		if player.x == enemy.x and player.y == enemy.y:
			reward = -1 * enemy_penalty

		elif player.x == food.x and player.y == food.y:
			reward = food_reward

		else:
			reward = -1 * move_penalty

		new_obs = (player-food, player-enemy)
		max_future_q = np.max(q_table[new_obs])
		current_q = q_table[obs][action]

		if reward == food_reward:
			new_q = food_reward

		elif reward == (-1 * enemy_penalty):
			new_q = -enemy_penalty

		else:
			new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

		q_table[obs][action] = new_q


		if show:
			env = np.zeros((size, size, 3), dtype=np.uint8) 
			env[food.x][food.y] = d[food_n]
			env[player.x][player.y] = d[player_n]
			env[enemy.x][enemy.y] = d[enemy_n]

			img = Image.fromarray(env, "RGB")
			img = img.resize((300,300))
			cv2.imshow("image",np.array(img))

			if reward == food_reward or reward == -enemy_penalty:
				if cv2.waitKey(500) & 0xFF == ord('q'):
					break
				else:
					if cv2.waitKey(2) & 0xFF == ord('q'):
						break

		episode_reward += reward

		if reward == food_reward or reward == -enemy_penalty :
			break

	episode_rewards.append(episode_rewards)
	epsilon *= eps_decay

moving_avg = np.convolve(episode_rewards, np.ones((show_every,)) / show_every, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {show_every}")
plt.xlabel("episode #")
plt.show()

with open(f"q-table-{time.time()}.pickle", "wb") as f:
	pickle.dump(q_table, f)




