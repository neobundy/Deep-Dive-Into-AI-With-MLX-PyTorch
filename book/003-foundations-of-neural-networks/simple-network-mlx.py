import mlx.core as mx
import mlx.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, num_neurons):
        super().__init__()
        self.layer = nn.Linear(input_size, num_neurons)

    def __call__(self, x):
        return self.layer(x)

# Create and initialize the network
input_size = 2
num_neurons = 3
model = SimpleNetwork(input_size, num_neurons)
mx.eval(model.parameters())

# Accessing parameters
params = model.parameters()

# Counting parameters
total_params = 0
for layer in params.values():
    for param in layer.values():
        total_params += param.size

print(f"Total number of parameters: {total_params}")
