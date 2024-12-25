from scipy.stats import poisson
# Probability of observing 4 arrivals in a time interval, with an average rate of 3 arrivals
print(poisson.pmf(k=4, mu=3))
