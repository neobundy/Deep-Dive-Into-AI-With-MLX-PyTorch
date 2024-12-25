import numpy as np

# Simplified example for dimension reduction

# Original matrix with 100 rows and 10,000 columns
pretrained_llm = np.random.rand(100, 10000)  # High-dimensional data

# Creating a projection matrix to reduce dimensions from 10,000 to 100
# This will be used to map the original data from the high-dimensional space (10,000 dimensions) to a lower-dimensional space (100 dimensions).
projection_matrix = np.random.rand(10000, 100)  # Transformation matrix

# Applying the projection to reduce dimensions
reduced_llm = np.dot(pretrained_llm, projection_matrix)  # Projected data

# Shape of the reduced matrix
print("Shape of the original matrix:", pretrained_llm.shape) # (100, 10000)
print("Shape of the reduced matrix:", reduced_llm.shape) # (100, 100)