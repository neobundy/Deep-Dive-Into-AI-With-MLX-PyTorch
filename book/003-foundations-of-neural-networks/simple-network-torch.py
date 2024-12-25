import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, num_neurons):
        super(SimpleNetwork, self).__init__()
        self.layer = nn.Linear(input_size, num_neurons)

    def forward(self, x):
        return self.layer(x)

# Creating the network
input_size = 2
num_neurons = 3
model = SimpleNetwork(input_size, num_neurons)

# Counting parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")