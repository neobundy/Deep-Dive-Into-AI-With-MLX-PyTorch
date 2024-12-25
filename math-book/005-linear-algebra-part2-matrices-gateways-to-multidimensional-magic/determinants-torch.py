import torch

# Define a 3x3 matrix
A = torch.tensor([[4.0, 2.0, 1.0],
                  [6.0, 3.0, 2.0],
                  [1.0, -1.0, 1.0]])

# Calculate the determinant
det_A = torch.det(A)

print("Matrix A:\n", A)
print("Determinant of A:", det_A)