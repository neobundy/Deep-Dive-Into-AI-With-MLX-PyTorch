import numpy as np

# Example data: actual and predicted values
y_actual = np.array([3, -0.5, 2, 7])
y_predicted = np.array([2.5, 0.0, 2, 8])

# Compute L1 loss
l1_loss = np.sum(np.abs(y_actual - y_predicted))

print("L1 Loss:", l1_loss)