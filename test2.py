import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

fig = plt.figure()
data = np.random.rand(10, 10)
sns.heatmap(data, vmax=.8, square=True)

def init():
      sns.heatmap(np.zeros((10, 1)), vmax=.8, square=False, cbar=False)

def animate(i):
    data = np.random.rand(10, 1)
    sns.heatmap(data, vmax=.8, square=False, cbar=False)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=20, repeat = False)
plt.show()