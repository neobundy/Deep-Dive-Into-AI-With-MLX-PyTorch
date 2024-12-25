import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Define dataset sizes
num_epochs = 10
n_total_samples = 7000  # Total number of samples
n_train_samples = int(n_total_samples * 0.7)  # 70% for training
n_val_samples = int(n_total_samples * 0.15)   # 15% for validation
n_test_samples = n_total_samples - n_train_samples - n_val_samples  # Remaining for testing

# Generate random features for the complete dataset
X = mx.random.uniform(shape=(n_total_samples, 10))  # 7000 samples, 10 features each

# Define a function to generate synthetic targets
def generate_target(X):
    noise = mx.random.normal(shape=(X.shape[0],)) * 0.1  # Adding some noise
    return mx.sum(X, axis=1) + noise

# Generate targets for the complete dataset
y = generate_target(X)

# Split the data into training, validation, and testing sets
X_train, X_val, X_test = mx.split(X, indices_or_sections=[n_train_samples, n_train_samples + n_val_samples])
y_train, y_val, y_test = mx.split(y, indices_or_sections=[n_train_samples, n_train_samples + n_val_samples])

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 1)

    def __call__(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        return x.squeeze()

# Create an instance of the model
model = RegressionModel()

# Define the loss function
def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)

# Get the loss and gradients function
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Define the optimizer
optimizer = optim.SGD(learning_rate=0.01)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for i in range(n_train_samples):
        input = mx.expand_dims(X_train[i], axis=0)
        target = y_train[i]
        loss, grads = loss_and_grad_fn(model, input, target)
        optimizer.update(model, grads)
        epoch_loss += loss.item()

    avg_loss = epoch_loss / n_train_samples

    # Validation loop
    model.eval()
    val_loss = 0.0
    for i in range(n_val_samples):
        input = mx.expand_dims(X_val[i], axis=0)
        target = y_val[i]
        val_loss += loss_fn(model, input, target).item()

    avg_val_loss = val_loss / n_val_samples
    print(f"Epoch {epoch}, Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}")

# Test the model
test_loss = 0.0
for i in range(n_test_samples):
    input = mx.expand_dims(X_test[i], axis=0)
    target = y_test[i]
    test_loss += loss_fn(model, input, target).item()
avg_test_loss = test_loss / n_test_samples
print(f"Test Loss: {avg_test_loss}")

# Define a prediction function
def predict(model, new_data):
    model.eval()
    new_data = mx.expand_dims(new_data, axis=0)
    predictions = model(new_data)
    return predictions.squeeze()

# Example usage
new_data = mx.array(X_test)  # Fill in with actual data
predictions = predict(model, new_data)
print(predictions)

