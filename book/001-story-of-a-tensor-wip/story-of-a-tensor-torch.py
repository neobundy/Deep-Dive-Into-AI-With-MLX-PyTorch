import torch
import torch.nn as nn
import torch.optim as optim

# Tenny is a tensor with random numbers
Tenny = torch.randn(10)

# Tenny reshapes himself into a vector, a matrix, and a 3D array
vector = Tenny.view(-1)
matrix = Tenny.view(5, 2)
three_d_array = Tenny.view(2, 5, 1)

# Define a simple linear regression model
class GreatNeuralNetwork(nn.Module):
    def __init__(self):
        super(GreatNeuralNetwork, self).__init__()
        self.linear = nn.Linear(1, 1)  # Weighy and Bitsy Bias are part of this

    def forward(self, x):
        x = self.linear(x)
        return x

model = GreatNeuralNetwork()

# Print Weighy and Bitsy Bias
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training data
x_train = torch.randn(10, 1)
y_train = 3*x_train + 2

# Training process
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Make a prediction
model.eval()
x_test = torch.tensor([[5.0]])
y_test = model(x_test)
print(f"Prediction: {y_test.item()}")