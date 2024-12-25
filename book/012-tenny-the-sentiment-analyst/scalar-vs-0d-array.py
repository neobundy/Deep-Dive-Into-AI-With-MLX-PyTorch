import numpy as np

# Creating a scalar
scalar_value = 5

# Creating a 0D array
array_0D = np.array(scalar_value)

# Displaying the scalar and the 0D array
print("Scalar Value:", scalar_value)
print("0D Array:", array_0D)

# Checking their types
print("Type of Scalar Value:", type(scalar_value))
print("Type of 0D Array:", type(array_0D))

# Trying to index the scalar and 0D array
try:
    print("Attempting to index Scalar Value:", scalar_value[0])
except TypeError as e:
    print("Error indexing Scalar Value:", e)

try:
    print("Attempting to index 0D Array:", array_0D[0])
except IndexError as e:
    print("Error indexing 0D Array:", e)
