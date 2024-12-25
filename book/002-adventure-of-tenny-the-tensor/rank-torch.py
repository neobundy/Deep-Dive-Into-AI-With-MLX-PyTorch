import torch

# Define matrices using PyTorch tensors: tensors should be floats or doubles
A = torch.tensor([[1, 2], [3, 6]], dtype=torch.float32)
B = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Calculate ranks using PyTorch's matrix_rank method
rank_A = torch.linalg.matrix_rank(A)
rank_B = torch.linalg.matrix_rank(B)

print("Rank of Matrix A:", rank_A.item())  # Output: 1
print("Rank of Matrix B:", rank_B.item())  # Output: 2
