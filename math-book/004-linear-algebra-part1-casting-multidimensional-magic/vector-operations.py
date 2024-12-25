import numpy as np

# Vector variables
v = np.array([1, 2])
w = np.array([3, 4])

# Addition
vector_addition = v + w
print("Vector Addition:", vector_addition)

# Division (Element-wise)
vector_division = v / w
print("Vector Division:", vector_division)

# Multiplication (Dot Product)
vector_multiplication = np.dot(v, w)
print("Vector Multiplication (Dot Product):", vector_multiplication)