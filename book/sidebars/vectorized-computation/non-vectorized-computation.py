import time

# Create a list with one million numbers
numbers = list(range(1, 1000001))

# Start timing
start_time = time.time()

# Square each number in the list
squared_numbers = [number ** 2 for number in numbers]

# End timing
end_time = time.time()

print("Execution Time (Non-Vectorized):", end_time - start_time, "seconds")