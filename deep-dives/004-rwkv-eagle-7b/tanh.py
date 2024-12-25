import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate a range of values from -10 to 10, which is a common range for inputs to tanh
x = np.linspace(-10, 10, 1000)
# Apply the tanh function to these values
y = np.tanh(x)

# Use seaborn to plot the function
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.lineplot(x=x, y=y, palette="tab10", linewidth=2.5)
plt.title('Hyperbolic Tangent Function - tanh')
plt.xlabel('Input Value (x)')
plt.ylabel('Output Value tanh(x)')
plt.show()
