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

# Original vectors
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='g', label='$\\vec{v} = [2,3]$')
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='b', label='$\\vec{u} = [4,0]$')

# Projection of v onto u
plt.quiver(0, 0, proj_u_v[0], proj_u_v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Projection of $\\vec{v}$ onto $\\vec{u}$')

# Correcting the line for perpendicular projection from v to the projection on u
perpendicular_end_point = [v[0], proj_u_v[1]]  # This will create a vertical line for the perpendicular
plt.plot([v[0], proj_u_v[0]], [v[1], proj_u_v[1]], 'k--', label='Perpendicular from $\\vec{v}$ to $\\vec{u}$')

# Set the limits of the plot
plt.xlim(-1, 5)
plt.ylim(-1, 5)

# Adding grid
plt.grid(True)

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Vector Projection with Perpendicular Concept')
plt.legend()

# Show the plot with seaborn style
sns.set()
plt.show()
