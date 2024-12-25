import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Creating a Venn diagram to visualize the probabilities
plt.figure(figsize=(15, 5))

# Joint Probability
plt.subplot(1, 3, 1)
venn2(subsets=(1, 1, 1), set_labels=('A', 'B'))
plt.title("Joint Probability: P(A ∩ B)")

# Marginal Probability
plt.subplot(1, 3, 2)
venn2(subsets=(1, 0, 0), set_labels=('A', 'B'))
plt.title("Marginal Probability: P(A) or P(B)")

# Conditional Probability
plt.subplot(1, 3, 3)
venn2(subsets=(0, 1, 1), set_labels=('A', 'B'))
plt.title("Conditional Probability: P(B|A)")

plt.tight_layout()
plt.show()