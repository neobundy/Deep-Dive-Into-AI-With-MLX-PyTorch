import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a skewed dataset: Exponential data is often used to demonstrate non-normality
np.random.seed(42) # Ensuring reproducibility
data = np.random.exponential(scale=2.0, size=1000)

# Applying a logarithmic transformation
log_data = np.log(data)

# Plotting the original and transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Original data plot
sns.histplot(data, kde=True, ax=ax1, color="skyblue")
ax1.set_title('Original Data')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

# Log-transformed data plot
sns.histplot(log_data, kde=True, ax=ax2, color="lightgreen")
ax2.set_title('Log-transformed Data')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')

plt.show()
