import matplotlib.pyplot as plt
import numpy as np

# Define the positions for cat, cute, and ball in a conceptual 2D space
positions = {
    'cat': np.array([2, 3]),
    'cute': np.array([2.5, 3.5]),
    'ball': np.array([7, 1])
}

# Create a scatter plot
plt.figure(figsize=(10, 6))
for word, pos in positions.items():
    plt.scatter(pos[0], pos[1], label=f'"{word}"', s=100)
    plt.text(pos[0], pos[1]+0.1, word, horizontalalignment='center', fontsize=12)

# Emphasize the proximity between 'cat' and 'cute' and distance to 'ball'
plt.plot([positions['cat'][0], positions['cute'][0]], [positions['cat'][1], positions['cute'][1]], 'g--')
plt.plot([positions['cat'][0], positions['ball'][0]], [positions['cat'][1], positions['ball'][1]], 'r:')

# Set the limit for the axes for better visualization
plt.xlim(0, 8)
plt.ylim(0, 5)

# Add title and legend
plt.title('Conceptual Vector Space Graph')
plt.legend()

# Display the plot
plt.show()
