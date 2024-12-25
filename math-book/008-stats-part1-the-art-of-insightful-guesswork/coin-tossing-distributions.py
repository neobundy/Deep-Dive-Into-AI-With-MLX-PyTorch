import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, uniform, norm

# Sample size
n = 1000

# Binomial Distribution (n=10, p=0.5)
binomial_data = binom.rvs(n=10, p=0.5, size=n)

# Bernoulli Distribution (p=0.5)
bernoulli_data = bernoulli.rvs(p=0.5, size=n)

# Uniform Distribution
uniform_data = uniform.rvs(size=n)

# Normal Distribution
normal_data = norm.rvs(size=n)

# Creating subplots
plt.figure(figsize=(20, 10))

# Binomial Distribution
plt.subplot(2, 2, 1)
sns.histplot(binomial_data, kde=False)
plt.title("Binomial Distribution")

# Bernoulli Distribution
plt.subplot(2, 2, 2)
sns.histplot(bernoulli_data, kde=False, discrete=True)
plt.title("Bernoulli Distribution")

# Uniform Distribution
plt.subplot(2, 2, 3)
sns.histplot(uniform_data, kde=True)
plt.title("Uniform Distribution")

# Normal Distribution
plt.subplot(2, 2, 4)
sns.histplot(normal_data, kde=True)
plt.title("Normal Distribution")

plt.tight_layout()
plt.show()