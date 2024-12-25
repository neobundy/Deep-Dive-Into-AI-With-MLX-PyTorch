import numpy as np

# Create a 0D array (scalar in array form)
scalar_in_array = np.array(5)

# Try to access an element in the 0D array
try:
    element = scalar_in_array[0]
    print("Element:", element)
except IndexError as e:
    print("Error accessing element in 0D array:", e)

# Try to iterate over the 0D array
try:
    for item in scalar_in_array:
        print("Iterating item:", item)
except TypeError as e:
    print("Error iterating over 0D array:", e)