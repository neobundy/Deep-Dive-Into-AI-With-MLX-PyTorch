import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Create a non-normal distribution: for example, an exponential distribution
data = np.random.exponential(scale=2.0, size=1000)

# Put the data into a Pandas DataFrame
df = pd.DataFrame(data, columns=['Non-Normal Data'])

# Normalize the data by applying a logarithmic transformation
df['Normalized Data'] = np.log(df['Non-Normal Data'])

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original non-normal data
sns.histplot(df['Non-Normal Data'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Non-Normal Distribution')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Plot the normalized data
sns.histplot(df['Normalized Data'], bins=30, kde=True, ax=axes[1], color='green')
axes[1].set_title('Normalized Distribution')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

# Display the plots
plt.tight_layout()
plt.show()
