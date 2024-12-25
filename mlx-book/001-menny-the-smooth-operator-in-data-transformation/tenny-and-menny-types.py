import torch
import mlx.core as mx
import numpy as np

python_vanilla_list = [1.0,2.0,3.0,4.0,5.0]
numpy_array = np.array(python_vanilla_list)

Tenny = torch.tensor(numpy_array)
Menny = mx.array(numpy_array)

print(python_vanilla_list, type(python_vanilla_list))
print(numpy_array, type(numpy_array))
print(Tenny, type(Tenny))
print(Menny, type(Menny))