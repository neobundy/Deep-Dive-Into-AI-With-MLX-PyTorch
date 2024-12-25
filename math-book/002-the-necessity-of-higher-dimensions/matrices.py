import numpy as np

# Define a simple 2x3 matrix using lists
simple_matrix = [
    [1, 2, 3],  # First row
    [4, 5, 6]   # Second row
]

# Accessing elements
print("First row:", simple_matrix[0])
print("Second row:", simple_matrix[1])
print("Element at row 2, column 3:", simple_matrix[1][2])

# Output:
# First row: [1, 2, 3]
# Second row: [4, 5, 6]
# Element at row 2, column 3: 6

# Creating a 2x3 matrix with NumPy
np_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Accessing elements with NumPy
print("NumPy Matrix:\n", np_matrix)
print("Element at row 2, column 3:", np_matrix[1, 2])

# Output:
# NumPy Matrix:
# [[1 2 3]
#  [4 5 6]]
# Element at row 2, column 3: 6
