from scipy.stats import binom
# Calculating the probability of 3 successful outcomes in 5 coin flips
print(binom.pmf(k=3, n=5, p=0.5))
