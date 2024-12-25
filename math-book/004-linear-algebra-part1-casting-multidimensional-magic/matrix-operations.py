import numpy as np

# Matrix variables
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Addition
matrix_addition = A + B
print("Matrix Addition:\n", matrix_addition)

# Division isn't typically defined for matrices in the same way as for scalars or element-wise for vectors.
# However, you can perform element-wise multiplication or use other specific operations like inverse, if needed.

# Multiplication
matrix_multiplication = np.dot(A, B)
print("Matrix Multiplication:\n", matrix_multiplication)