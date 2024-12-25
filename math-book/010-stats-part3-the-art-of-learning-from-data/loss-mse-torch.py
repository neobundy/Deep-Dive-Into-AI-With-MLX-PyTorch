import torch
import torch.nn as nn

# Define a simple dataset
X = torch.tensor([[1.0], [2.0], [3.0]]) # Input features
y = torch.tensor([[2.0], [4.0], [6.0]]) # Actual outputs

# Initialize a linear regression model
model = nn.Linear(in_features=1, out_features=1)

# Specify the Mean Squared Error Loss function
loss_fn = nn.MSELoss()

# Perform a forward pass to get the model's predictions
predictions = model(X)

# Calculate the MSE loss between the model's predictions and the actual values
mse_loss = loss_fn(predictions, y)

print(f"MSE Loss: {mse_loss.item()}")