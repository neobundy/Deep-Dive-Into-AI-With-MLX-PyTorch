import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the model as a class
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# Instantiate the model
model = NeuralNet()

# Generate some random data
input_data = torch.randn(1, 10)

# Forward pass
output = model(input_data)
print(output)