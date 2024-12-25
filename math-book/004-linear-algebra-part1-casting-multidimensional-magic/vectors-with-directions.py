import matplotlib.pyplot as plt
import seaborn as sns

# Setting the visual stage for exploring direction
fig, ax = plt.subplots()

# Defining vectors for various disciplines
vectors_disciplines = {
    'Physics & Engineering': (0, 0, 3, 2),
    'Navigation & Geography': (0, 0, -2, 3),
    'Mathematics': (0, 0, 4, -1),
    'Computer Graphics & Vision': (0, 0, -1, -3),
    'Biology & Chemistry': (0, 0, 2, -2)
}

# Assigning a unique color to each vector
colors = sns.color_palette('husl', n_colors=len(vectors_disciplines))

# Plotting each vector with a label
for (label, vector), color in zip(vectors_disciplines.items(), colors):
    ax.quiver(*vector, angles='xy', scale_units='xy', scale=1, color=color, label=label)

# Customizing the visual field
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Significance of Direction in Various Disciplines')

# Incorporating a legend to guide interpretation
ax.legend()

# Revealing the plotted insights
plt.grid(True)
plt.show()