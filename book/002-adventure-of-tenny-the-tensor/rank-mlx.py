import mlx.core as mx

# As of 0.0.7 mlx lacks a rank function

# Define matrices
A = mx.array([[1, 2], [3, 6]], dtype=mx.float32)
B = mx.array([[1, 2], [3, 4]], dtype=mx.float32)

# Function to compute the rank of a 2x2 matrix
def rank_2x2(matrix):
    # Check for zero matrix
    if mx.equal(matrix, mx.zeros_like(matrix)).all():
        return 0
    # Check for determinant equals zero for non-invertible matrix
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if det == 0:
        return 1
    # Otherwise, the matrix is invertible (full rank)
    return 2

# Calculate ranks
rank_A = rank_2x2(A)
rank_B = rank_2x2(B)

print("Rank of Matrix A:", rank_A)  # Output should be 1
print("Rank of Matrix B:", rank_B)  # Output should be 2