import torch
import torch.nn as nn
import torch.optim as optim

# Simple Neural Network
class ArcheryModel(nn.Module):
    def __init__(self):
        super(ArcheryModel, self).__init__()
        self.layer = nn.Linear(1, 1)  # Single input to single output
        self.activation = nn.ReLU()   # ReLU Activation Function

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

# Instantiate the model
model = ArcheryModel()

# Loss Function
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example training data: [input, target]
train_data = [(torch.tensor([0.5]), torch.tensor([0.8]))]

# Training Loop
for epoch in range(100):  # Number of times to iterate over the dataset
    for input, target in train_data:
        # Forward pass
        output = model(input)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")