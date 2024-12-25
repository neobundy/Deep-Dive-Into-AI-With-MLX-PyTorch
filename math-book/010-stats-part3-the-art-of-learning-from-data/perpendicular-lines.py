import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generating synthetic data
np.random.seed(0)
x = np.random.rand(10, 1) * 10
y = 2 * x + 1 + np.random.randn(10, 1) * 2

# Fitting a linear regression model
model = LinearRegression()
model.fit(x, y)
x_fit = np.linspace(0, 10, 100).reshape(-1, 1)
y_fit = model.predict(x_fit)

# Plotting the data and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x_fit, y_fit, color='red', label='Regression Line')

# Correcting the plotting of the loss lines
for xi, yi in zip(x, y):
    y_line = model.predict(xi.reshape(-1, 1))
    # Make sure y_line is a scalar by indexing the result of model.predict
    plt.plot([xi, xi], [yi, y_line[0]], color='green', linestyle='--')  # Corrected line

plt.title('Linear Regression with Perpendicular Loss Lines')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.legend()
plt.show()
