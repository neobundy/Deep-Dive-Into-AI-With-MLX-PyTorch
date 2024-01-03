import torch
import torch.nn as nn

# Tenny is a tensor with random numbers
Tenny = torch.randn(10)

# Tenny reshapes himself into a vector, a matrix, and a 3D array
vector = Tenny.reshape(-1)
matrix = Tenny.reshape(5, 2)
three_d_array = Tenny.reshape(2, 5, 1)

# Define a simple linear regression model
class GreatNeuralNetwork(nn.Module):
    def __init__(self):
        super(GreatNeuralNetwork, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)  # Weighy and Bitsy Bias are part of this

    def __call__(self, x):
        x = self.linear(x)
        return x

model = GreatNeuralNetwork()

# Print Weighy and Bitsy Bias

for name, param in model.named_parameters():
    print(name, param)

# Define a loss function and an optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training data
x_train = torch.randn(10, 1)
y_train = 3*x_train + 2

# Training process
for epoch in range(100):
    predictions = model(x_train)
    loss = loss_function(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make a prediction
x_test = torch.tensor([[5.0]])
y_test = model(x_test)
print(f"Prediction: {y_test.item()}")
