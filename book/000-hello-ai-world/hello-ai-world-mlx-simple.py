import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

x_train = [40, 1, 16, 95, 79, 96, 9, 35, 37, 63, 45, 98, 75, 48, 25, 90, 27, 71, 35, 32]
y_train = [403, 13, 163, 953, 793, 963, 93, 353, 373, 633, 453, 983, 753, 483, 253, 903, 273, 713, 353, 323]

x_train_tensor = mx.array(x_train, dtype=mx.float32).reshape(-1, 1)
y_train_tensor = mx.array(y_train, dtype=mx.float32).reshape(-1, 1)

model = nn.Linear(input_dims=1, output_dims=1)
mx.eval(model.parameters())

learning_rate = 0.0001

def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=learning_rate)

num_epochs = 5000

for epoch in range(num_epochs):
    loss, grads = loss_and_grad_fn(model, x_train_tensor, y_train_tensor)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.5f}')

input_number = mx.array([100.0]).reshape(-1, 1)
predicted_output = model(input_number)
print(f'For input x = 100, the predicted y value is: {predicted_output.item()}')
