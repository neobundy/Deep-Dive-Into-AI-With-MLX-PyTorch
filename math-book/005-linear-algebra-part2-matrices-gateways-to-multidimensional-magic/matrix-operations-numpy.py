import numpy as np

# Define two matrices of the same dimensions
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Add the matrices
C = A + B

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Addition (A + B):\n", C)

# Subtract B from A
D = A - B

print("Subtraction (A - B):\n", D)

# Define a scalar value
scalar = 2

# Scale matrix A by the scalar
E = A * scalar

print("Matrix A:\n", A)
print("Scalar:", scalar)
print("Scaling (A * scalar):\n", E)