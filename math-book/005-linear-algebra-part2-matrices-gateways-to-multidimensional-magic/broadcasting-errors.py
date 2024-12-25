import numpy as np

# Create a 2x2 matrix
matrix = np.array([[1, 2], [3, 4]])

# Define a 3-element vector
vector_3 = np.array([1, 2, 3])

# Attempt to add it to our 2x2 matrix
try:
    result = matrix + vector_3
    print("Result:\n", result)
except ValueError as e:
    print("Broadcasting Error due to shape mismatch:", e)