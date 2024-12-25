import math

# Number of heads, total tosses, and probability of heads
k, n, p = 7, 10, 0.5

# Calculate the log likelihood
log_likelihood = k * math.log(p) + (n - k) * math.log(1 - p)
print("Log Likelihood:", log_likelihood)
# Log Likelihood: -6.931471805599452