import numpy as np

# Example data: actual and predicted values
y_actual = np.array([3, -0.5, 2, 7])
y_predicted = np.array([2.5, 0.0, 2, 8])

# Compute L2 loss
l2_loss = np.sum((y_actual - y_predicted) ** 2)

print("L2 Loss:", l2_loss)