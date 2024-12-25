import numpy as np

# Sample data
data = np.array([2, 3, 5, 7, 11])

# Estimating parameters
mean_estimate = np.mean(data)
std_dev_estimate = np.std(data, ddof=1)

print("Estimated Mean:", mean_estimate)
print("Estimated Standard Deviation:", std_dev_estimate)