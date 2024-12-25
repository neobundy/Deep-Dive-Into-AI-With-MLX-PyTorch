import numpy as np

# Define the coefficient matrix A and the constant vector B
A = np.array([[2, 3], [4, -1]])
B = np.array([5, 3])

# Solve for X
X = np.linalg.solve(A, B)

print("Solution using NumPy:", X)
