import torch

# Define the matrix A
A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)