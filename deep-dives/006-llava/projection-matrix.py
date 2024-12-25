import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the vectors v and u
v = np.array([2, 3])
u = np.array([4, 0])

# Calculate the projection of v onto u
# proj_u_v = (dot(v, u) / dot(u, u)) * u
proj_u_v = (np.dot(v, u) / np.dot(u, u)) * u

# Plotting the vectors and their projection
plt.figure(figsize=(10,6))

# Plot the vector v
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='g', label='v [2,3]')

# Plot the vector u
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='b', label='u [4,0]')

# Plot the projection of v onto u
plt.quiver(0, 0, proj_u_v[0], proj_u_v[1], angles='xy', scale_units='xy', scale=1, color='r', label='proj_u_v [2,0]')

# Set the limits of the plot
plt.xlim(-1, 5)
plt.ylim(-1, 5)

# Adding grid
plt.grid(True)

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Vector Projection')
plt.legend()

# Show the plot with seaborn style
sns.set()
plt.show()
