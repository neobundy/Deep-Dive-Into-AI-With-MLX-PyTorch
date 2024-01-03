import mlx.core as mx
import mlx.nn as nn


# Define the model as a class
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Linear(10, 20),
            nn.Linear(20, 2)
        ]

    def __call__(self, x):
        for i, l in enumerate(self.layers):
            # Apply ReLU to all but the last layer
            x = mx.maximum(x, 0) if i > 0 else x
            x = l(x)
        return x


# Instantiate the model
model = NeuralNet()

# Generate some random data
input_data = mx.random.normal((1,10))

# Forward pass
output = model(input_data)
print(output)