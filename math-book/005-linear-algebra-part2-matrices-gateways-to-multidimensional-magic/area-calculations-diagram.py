import matplotlib.pyplot as plt
import numpy as np

# Define the origin
origin = np.array([0, 0])

# Define vectors u and v
u = np.array([3, 5])
v = np.array([2, 7])

# Calculate the fourth vertex of the parallelogram
fourth_vertex = u + v

# Plotting the parallelogram
plt.figure(figsize=(8, 6))
plt.quiver(*origin, *u, scale=1, scale_units='xy', angles='xy', color='r', label='Vector u (3,5)')
plt.quiver(*origin, *v, scale=1, scale_units='xy', angles='xy', color='g', label='Vector v (2,7)')
plt.quiver(*u, *v, scale=1, scale_units='xy', angles='xy', color='g')
plt.quiver(*v, *u, scale=1, scale_units='xy', angles='xy', color='r')

# Plotting the fourth vertex to complete the parallelogram
plt.plot([u[0], fourth_vertex[0]], [u[1], fourth_vertex[1]], 'r--')
plt.plot([v[0], fourth_vertex[0]], [v[1], fourth_vertex[1]], 'g--')

# Setting the plot limits
plt.xlim(-1, 10)
plt.ylim(-1, 15)

# Adding labels and legend
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title('Parallelogram Formed by Vectors u and v')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()
