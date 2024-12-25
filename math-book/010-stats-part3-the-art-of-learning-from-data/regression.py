import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data for a simple linear relationship
np.random.seed(42) # For reproducibility
x = np.arange(0, 100, 5)
y = 2 * x + np.random.normal(0, 10, len(x)) # Linear relationship with some noise

# Plotting the data and a linear regression model fit
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y)
plt.title('Linear Regression: Good Fit')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.show()
