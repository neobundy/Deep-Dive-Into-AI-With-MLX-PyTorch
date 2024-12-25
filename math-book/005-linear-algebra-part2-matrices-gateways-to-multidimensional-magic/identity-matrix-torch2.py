import torch

# Creating a 3x3 identity matrix
I = torch.eye(3)
print("Identity Matrix:\n", I)

# Creating an arbitrary 3x3 matrix A
A = torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=torch.float)
print("Arbitrary Matrix A:\n", A)

# Multiplying A by the identity matrix I
AI = torch.mm(A, I)
print("Result of A multiplied by I:\n", AI)

# Verifying if AI is equal to A
is_equal = torch.equal(AI, A)
print("Is AI equal to A?:", is_equal)