import numpy as np

# Set a fixed seed for reproducibility
np.random.seed(42)

# Create arrays from 0D to 10D
arrays = {}

# Generating a 0D array with a random integer
arrays['0D'] = np.array(np.random.randint(0, 10))

# For higher dimensions, we will use tuple unpacking with `np.random.randint`
for i in range(1, 11):
    # Creating a shape tuple for the current dimension (i.e., i-dimensional shape)
    shape = tuple([2] * i)  # Using 2 for simplicity, but this can be any size
    arrays[f'{i}D'] = np.random.randint(0, 10, size=shape)

# Print out the dimension and shape for each array
for dim, array in arrays.items():
    print(f"{dim} array (ndim: {array.ndim}, shape: {array.shape}):")
    print(array) # Uncomment this line to see the array
    print()  # Empty line for readability