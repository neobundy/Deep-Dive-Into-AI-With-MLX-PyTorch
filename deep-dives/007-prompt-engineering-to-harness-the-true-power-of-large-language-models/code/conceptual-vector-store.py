import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define terms and their positions in a hypothetical 2D vector space
terms = ['apple', 'orange', 'lemon', 'car', 'cat']
# Hypothetical coordinates for these terms in a 2D space
coordinates = np.array([
    [5, 5],  # apple
    [4, 5],  # orange
    [4, 4],  # lemon
    [1, 1],  # car
    [1, 2]   # cat
])

# Create a DataFrame
df = pd.DataFrame(coordinates, index=terms, columns=['X', 'Y'])

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='X', y='Y', s=100)

for term in terms:
    plt.text(df.loc[term, 'X'] + 0.1, df.loc[term, 'Y'], term, fontsize=12)

plt.title('Conceptual Vector Space of Terms')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
