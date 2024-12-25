import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up the data for the graph
x = np.linspace(-10, 10, 400)
m = 2  # Slope
b = -5  # y-intercept
y = m * x + b  # Linear equation

# Plotting using seaborn
plt.figure(figsize=(8, 6))
sns.lineplot(x=x, y=y)
plt.title('Graph of y = mx + b with m=2 and b=-5')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
