import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class ArcheryModel(nn.Module):
    def __init__(self):
        super(ArcheryModel, self).__init__()
        self.layer = nn.Linear(1, 1)  # Single input to single output
        self.activation = nn.ReLU()  # ReLU Activation Function

    def __call__(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x


model = ArcheryModel()


def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

optimizer = optim.SGD(learning_rate=0.01)

train_data = [(mx.array([0.5]), mx.array([0.8]))]

for epoch in range(100):  # Number of times to iterate over the dataset
    for input, target in train_data:
        # Forward pass and loss calculation
        loss, grads = loss_and_grad_fn(model, input, target)

        # Backward pass and optimization
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        print(f"Epoch {epoch}, Loss: {loss.item()}")