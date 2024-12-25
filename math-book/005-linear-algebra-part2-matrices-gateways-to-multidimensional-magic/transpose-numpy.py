import numpy as np

# Define a 2x3 matrix
A = np.array([[1, 2, 3], [4, 5, 6]])

# Transpose the matrix using the .T attribute
A_transposed = A.T

print("Original Matrix:\n", A)
print("Transposed Matrix:\n", A_transposed)