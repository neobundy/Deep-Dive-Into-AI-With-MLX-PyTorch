import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setting Seaborn style for aesthetic enhancement
sns.set()

# Establishing the canvas
fig, ax = plt.subplots()

# Defining example vectors
vectors = [(0, 0, 2, 3), (0, 0, -1, -1), (0, 0, 4, 1)]

# Plotting each vector
for vector in vectors:
    ax.quiver(*vector, angles='xy', scale_units='xy', scale=1, color=np.random.rand(3,))

# Configuring the stage
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Graph of Vectors')

# Bringing the plot to life
plt.grid(True)
plt.show()