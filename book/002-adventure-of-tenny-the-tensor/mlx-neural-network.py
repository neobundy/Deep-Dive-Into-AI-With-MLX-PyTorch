# Importing necessary libraries from MLX
import mlx.core as mx
import mlx.nn as nn

# Defining a Neural Network class
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Here, we're creating layers of the neural network.
        # Think of each layer as a stage in Tenny's learning journey.
        self.layers = [
            nn.Linear(10, 20),  # The first layer transforms input from 10 units to 20.
            nn.Linear(20, 2)    # The second layer changes those 20 units into 2.
        ]

    # This is what happens when data passes through Tenny (the model)
    def __call__(self, x):
        # x is the input data that Tenny is going to learn from.
        for i, l in enumerate(self.layers):
            # Tenny processes the input through each layer.
            # For all but the last layer, we use a function called ReLU,
            # which helps Tenny make better decisions.
            x = mx.maximum(x, 0) if i > 0 else x
            x = l(x)  # Tenny takes the output of one layer and uses it as input for the next.
        return x  # After passing through all layers, Tenny gives us the final output.

# Creating an instance of the Neural Network, which is Tenny starting its journey.
model = NeuralNet()