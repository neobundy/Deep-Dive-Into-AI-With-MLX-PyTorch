import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate random data for 30 students' performance, assuming a somewhat normal distribution
np.random.seed(42) # For reproducibility
student_performance = np.random.normal(loc=75, scale=10, size=30) # loc is the mean, scale is the standard deviation

# Create the histogram with a smoothing line for the normal distribution
plt.figure(figsize=(10, 6))
sns.histplot(student_performance, bins=10, kde=True, color="skyblue")
plt.title('Histogram of 30 Students\' Performance with Normal Distribution Curve')
plt.xlabel('Performance Score')
plt.ylabel('Number of Students')
plt.grid(True)
plt.show()
