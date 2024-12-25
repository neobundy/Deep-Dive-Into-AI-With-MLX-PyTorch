import matplotlib.pyplot as plt
import seaborn as sns

# Creating a new figure and axis for the 'What the Heck Is Direction?' section
fig, ax = plt.subplots()

# Example vectors representing different disciplines
vectors_disciplines = {
    'Physics & Engineering': (0, 0, 3, 2),
    'Navigation & Geography': (0, 0, -2, 3),
    'Mathematics': (0, 0, 4, -1),
    'Computer Graphics & Vision': (0, 0, -1, -3),
    'Biology & Chemistry': (0, 0, 2, -2)
}

# Colors for different vectors
colors = sns.color_palette('husl', n_colors=len(vectors_disciplines))

# Adding vectors to the plot with labels
for (label, vector), color in zip(vectors_disciplines.items(), colors):
    ax.quiver(*vector, angles='xy', scale_units='xy', scale=1, color=color, label=label)

# Setting the limits, labels, and title
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Significance of Direction in Various Disciplines')

# Adding a legend
ax.legend()

# Display the plot
plt.grid(True)
plt.show()
