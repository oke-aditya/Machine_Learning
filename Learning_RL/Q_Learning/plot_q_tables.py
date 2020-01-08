from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os

style.use('ggplot')
folder_path = "E:\\Aditya\\AI_DL_ML\\Machine_Learning\\Learning_RL\\Q_Learning"

i = 200
q_table_file = f"qtables{i}.npy"
q_table_path = os.path.join(folder_path, q_table_file)

def get_q_color(value, vals):
	if value == max(vals):
		return "green", 1.0
	else:
		return "red", 0.3

fig = plt.figure(figsize = (12, 9))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

q_table = np.load(q_table_path)

for x, x_vals in enumerate(q_table):
    for y, y_vals in enumerate(x_vals):
        ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
        ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
        ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

        ax1.set_ylabel("Action 0")
        ax2.set_ylabel("Action 1")
        ax3.set_ylabel("Action 2")


plt.show()
i = 24999