from math import comb

# Total number of tosses (n), number of heads observed (k), probability of heads for a fair coin (p)
n, k, p = 10, 7, 0.5

# Calculate the likelihood
likelihood = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
print("Likelihood:", likelihood)
# Likelihood: 0.1171875