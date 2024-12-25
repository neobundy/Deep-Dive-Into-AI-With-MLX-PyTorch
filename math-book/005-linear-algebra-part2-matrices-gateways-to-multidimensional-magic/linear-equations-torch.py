import torch

# Define the coefficient matrix A and the constant vector B
A = torch.tensor([[2.0, 3.0], [4.0, -1.0]])
B = torch.tensor([5.0, 3.0])

# Solve for X
X = torch.linalg.solve(A, B)

print("Solution using PyTorch:", X)
