import numpy as np

# Define a 2x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Add a scalar value
result = matrix + 10

print("Result:\n", result)

# Define a one-dimensional array
vector = np.array([1, 0, -1])

# Add the vector to the matrix
result_with_vector = matrix + vector

print("Result with Vector:\n", result_with_vector)

# Define a column vector
column_vector = np.array([[1], [0]])

# Add the column vector to the matrix
result_with_column_vector = matrix + column_vector

print("Result with Column Vector:\n", result_with_column_vector)