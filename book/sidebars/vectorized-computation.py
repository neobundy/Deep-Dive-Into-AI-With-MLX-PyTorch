import numpy as np
import time

# Create a NumPy array with one million numbers
numbers = np.arange(1, 1000001)

# Start timing
start_time = time.time()

# Square each number in the NumPy array
squared_numbers = numbers ** 2

# End timing
end_time = time.time()

print("Execution Time (Vectorized):", end_time - start_time, "seconds")
