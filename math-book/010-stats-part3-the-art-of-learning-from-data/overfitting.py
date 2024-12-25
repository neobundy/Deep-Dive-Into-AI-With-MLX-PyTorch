from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Generate the same synthetic data
x = np.arange(0, 100, 5)
y = 2 * x + np.random.normal(0, 10, len(x))

# Transforming the data to fit a polynomial regression model
x_reshape = x[:, np.newaxis] # Reshaping for compatibility with model
model = make_pipeline(PolynomialFeatures(degree=10), LinearRegression())
model.fit(x_reshape, y)
y_pred = model.predict(x_reshape)

# Plotting the original data and the overfitted model predictions
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='Overfitted Model')
plt.title('Polynomial Regression: Overfitting')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.legend()
plt.show()
