import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Tenny is a tensor with random numbers
Tenny = mx.random.normal(shape=[10])

# Tenny reshapes himself into a vector, a matrix, and a 3D array
vector = Tenny.reshape(-1)
matrix = Tenny.reshape(5, 2)
three_d_array = Tenny.reshape(2, 5, 1)

# Define a simple linear regression model
class GreatNeuralNetwork(nn.Module):
    def __init__(self):
        super(GreatNeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_dims=1, output_dims=1)  # Weighy and Bitsy Bias are part of this

    def __call__(self, x):
        x = self.linear(x)
        return x

model = GreatNeuralNetwork()

# Print Weighy and Bitsy Bias
for name, param in model.parameters().items():
    print(name, param)

# Define a loss function and an optimizer
def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=0.01)

# Training data
x_train = mx.random.normal([10, 1])
y_train = 3*x_train + 2

# Training process
for epoch in range(100):
    loss, grads = loss_and_grad_fn(model, x_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

# Make a prediction
x_test = mx.array([[5.0]])
y_test = model(x_test)
print(f"Prediction: {y_test.item()}")