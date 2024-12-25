import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setting Seaborn style
sns.set()

# Creating a figure and axis
fig, ax = plt.subplots()

# Example vectors
vectors = [(0, 0, 2, 3), (0, 0, -1, -1), (0, 0, 4, 1)]

# Adding vectors to the plot
for vector in vectors:
    ax.quiver(*vector, angles='xy', scale_units='xy', scale=1, color=np.random.rand(3,))

# Setting the limits and labels
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Graph of Vectors')

# Display the plot
plt.grid(True)
plt.show()
