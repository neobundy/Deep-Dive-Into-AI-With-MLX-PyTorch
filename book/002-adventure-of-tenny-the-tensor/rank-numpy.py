import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 6]])
B = np.array([[1, 2], [3, 4]])

# Calculate ranks
rank_A = np.linalg.matrix_rank(A)
rank_B = np.linalg.matrix_rank(B)

print("Rank of Matrix A:", rank_A)  # Output: 1
print("Rank of Matrix B:", rank_B)  # Output: 2