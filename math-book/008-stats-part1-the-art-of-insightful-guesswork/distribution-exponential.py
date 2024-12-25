from scipy.stats import expon

# Define the rate parameter λ
lambda_ = 3  # Example value

# Calculate the probability density at x=1, given the rate parameter λ
print(expon.pdf(x=1, scale=1/lambda_))  # λ is the rate parameter
