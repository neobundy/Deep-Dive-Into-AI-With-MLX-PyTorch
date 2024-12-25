import torch

# Creating a Torch tensor
torch_tensor = torch.tensor([1, 2, 3, 4, 5])

# Vectorized multiplication
torch_result = torch_tensor * 2
print(torch_result)
# tensor([ 2,  4,  6,  8, 10])