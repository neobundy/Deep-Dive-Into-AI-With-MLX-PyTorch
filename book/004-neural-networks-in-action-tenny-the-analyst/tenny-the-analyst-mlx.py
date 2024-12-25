import os
import pandas as pd
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from sklearn.preprocessing import StandardScaler

# Define paths and hyperparameters for the ML process
folder_path = './data'                   # Path to the directory with CSV files for training.
prediction_data_path = './new_data/raw_data-nflx.csv'  # Path to new data for making predictions.
# prediction_data_path = './new_data/raw_data-avgo.csv'  # Path to new data for making predictions.
num_epochs = 5000                        # Total number of training iterations over the dataset.
batch_size = 100                         # Number of samples per batch to load.
hidden_size = 30                         # Number of units in hidden layers of the neural network.
output_size = 1                          # Number of units in the output layer (target prediction).
learning_rate = 0.0001                   # Step size at each iteration while moving toward a minimum of the loss function.
train_ratio = 0.7                        # Proportion of dataset to include in the training split.
val_ratio = 0.2                          # Proportion of dataset to include in the validation split.

# Helper function to convert data to tensors
# Convert features and labels to tensors
def tensors_to_device(features, labels):
    features_tensor = mx.array(features)
    labels_tensor = mx.expand_dims(mx.array(labels), axis=1)  # Labels need to be a 2D tensor
    return features_tensor, labels_tensor

# Split the companies
def split_data(file_names, train_ratio, val_ratio):
    total_files = len(file_names)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    train_files = file_names[:train_size]
    val_files = file_names[train_size:train_size + val_size]
    test_files = file_names[train_size + val_size:]

    return train_files, val_files, test_files


# Function to clean data
def clean_data(df):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_cleaned = df.copy()  # Work on this copy to ensure we're not modifying a slice

    # Replace non-numeric placeholders with NaN
    df_cleaned.replace(['#VALUE!', '-'], pd.NA, inplace=True)

    # Ensure all data is numeric
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values in numerical columns with column mean
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype == 'float64' or df_cleaned[column].dtype == 'int64':
            df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)

    return df_cleaned


# Function to read and clean data from files
def read_and_clean_data(files):
    data = pd.DataFrame()
    for file in files:
        file_path = os.path.join(folder_path, file)
        temp_df = pd.read_csv(file_path, index_col=0)
        temp_df = temp_df.transpose()  # Transpose the data
        temp_df = clean_data(temp_df)  # Clean the data

        # Concatenate to the main dataframe
        data = pd.concat([data, temp_df], ignore_index=True)

    data = pd.DataFrame(data)  # Convert back to DataFrame if needed
    data.fillna(data.mean(), inplace=True)
    return data


def prepare_features_labels(data_df):
    # Assuming 'data_df' is already read, transposed, and cleaned
    # The first column is the label, and the rest are features

    # Extract features and labels
    features = data_df.iloc[:-1, 1:]  # all rows except the last, all columns except the first
    labels = data_df.iloc[1:, 0]  # all rows from the second, only the first column as labels

    # Convert to numpy arrays if not already and return
    return features.values, labels.values


# Create mini-batches from the data
def batchify_data(features, labels, batch_size, shuffle=True):
    dataset_size = labels.size
    indices = mx.arange(dataset_size)
    if shuffle:
        # Using numpy for permutation and then converting to mlx array
        # This step is necessary since we don't have mx.random.permutation yet
        indices = mx.array(np.random.permutation(dataset_size))

    for start_idx in range(0, dataset_size, batch_size):
        # Using slice notation to create batches
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]

        # Indexing with the mlx array of indices
        yield features[batch_indices], labels[batch_indices]

# Define the neural network architecture
class Tenny(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Tenny, self).__init__()
        # Define linear layers and a dropout layer for regularization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  # To prevent overfitting
        self.relu = nn.ReLU()

    def __call__(self, x):
        # Apply ReLU activations to linear layers and include dropout after the second layer
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Load data, preprocess with scaling, and split into training, validation, and test sets
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

train_files, val_files, test_files = split_data(file_names, train_ratio, val_ratio)

train_data = read_and_clean_data(train_files)
val_data = read_and_clean_data(val_files)
test_data = read_and_clean_data(test_files)

# Define feature columns and scaler
features_columns = train_data.columns[1:]  # Assuming first column is the label
scaler = StandardScaler()

# Scale the features of the train dataset and keep the labels aside
scaler.fit(train_data[features_columns])
train_features_scaled = scaler.transform(train_data[features_columns])
train_labels = train_data.iloc[:, 0].values  # Labels are the first column

# Apply the scaling to validation and test datasets
val_features_scaled = scaler.transform(val_data[features_columns])
val_labels = val_data.iloc[:, 0].values

test_features_scaled = scaler.transform(test_data[features_columns])
test_labels = test_data.iloc[:, 0].values

# Instead of creating TensorDataset and DataLoader, we use the batchify_data function directly
# Convert the scaled features and labels to tensors
train_features_tensor, train_labels_tensor = mx.array(train_features_scaled), mx.array(train_labels)
val_features_tensor, val_labels_tensor = mx.array(val_features_scaled), mx.array(val_labels)
test_features_tensor, test_labels_tensor = mx.array(test_features_scaled), mx.array(test_labels)

# Initialize neural network model
input_size = train_features_tensor.shape[1]  # Determine input size from the training dataset
model = Tenny(input_size=len(features_columns), hidden_size=hidden_size, output_size=output_size)

# Define loss function and optimizer
def loss_fn(model, x, y):
    y_pred = model(x)
    return nn.losses.mse_loss(y_pred, mx.expand_dims(y, axis=1))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=learning_rate)

best_val_loss = float('inf')  # Initialize best validation loss
patience = 10
no_improve = 0

# The model is created with all its parameters but nothing is initialized: MLX is lazily evaluated
# We force evaluate all parameters to initialize Tenny
mx.eval(model.parameters())

# Train the neural network
for epoch in range(num_epochs):
    for inputs, labels in batchify_data(train_features_tensor, train_labels_tensor, batch_size, shuffle=True):
        loss, grads = loss_and_grad_fn(model, inputs, labels)
        optimizer.update(model, grads)
        # Check for NaN in loss value to prevent invalid computations
        # Inside your training loop where loss is calculated:
        # Force a graph evaluation
        mx.eval(model.parameters(), optimizer.state)
        loss_value = loss.item()  # Convert MLX array to Python scalar
        if np.isnan(loss_value):
            print(f"NaN detected in loss at epoch {epoch + 1}")
            break

    # Validation phase
    val_loss = 0
    num_batches = 0
    # Manually handle batch iteration since MLX doesn't have DataLoader
    for inputs, labels in batchify_data(val_features_tensor, val_labels_tensor, batch_size, shuffle=False):
        # Forward pass and loss calculation. In MLX, .item() is not needed; loss is a scalar array by default.
        val_loss += loss_fn(model, inputs, labels)
        num_batches += 1
        mx.eval(model.parameters(), optimizer.state)  # Trigger computation for validation loss

    # Ensure num_batches is not zero to avoid division by zero
    if num_batches > 0:
        val_loss /= num_batches  # Calculate the average loss over the validation set
    else:
        # If there's no data, handle it. This should not normally occur.
        val_loss = mx.array([0.0])

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    # Early stopping logic...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        # Save model code would have to be added manually if needed
    else:
        no_improve += 1
        if no_improve == patience:
            print("No improvement in validation loss for {} epochs, stopping training.".format(patience))
            break

# Evaluate the model on the test dataset
test_loss = 0
num_batches = 0
# Manually handle batch iteration since MLX doesn't have DataLoader
for inputs, labels in batchify_data(test_features_tensor, test_labels_tensor, batch_size, shuffle=False):
    # Forward pass and loss calculation. In MLX, .item() is not needed; loss is a scalar array by default.
    test_loss += loss_fn(model, inputs, labels)
    num_batches += 1
    mx.eval(model.parameters(), optimizer.state)  # Trigger computation for validation loss
# Ensure num_batches is not zero to avoid division by zero
if num_batches > 0:
    test_loss /= num_batches  # Calculate the average loss over the test set
else:
    # If there's no data, handle it. This should not normally occur.
    test_loss = mx.array([0.0])

# Now test_loss is an MLX array, and we can get its scalar value for output
print(f"Average Test Loss: {test_loss.item():.4f}")

do_prediction = True

if do_prediction:

    # Process new data for prediction with proper reshaping
    new_data_df = pd.read_csv(prediction_data_path, index_col=0)
    new_data_df = new_data_df.transpose()  # Transpose it to align with the training data orientation

    # Clean the new data using the same function applied during training
    cleaned_new_data_df = clean_data(new_data_df)

    # Clean and preprocess the data using the same steps as for the training data
    new_data_features = cleaned_new_data_df.iloc[:, 1:]  # The first column is the label
    cleaned_new_features = clean_data(new_data_features)
    scaled_new_features = scaler.transform(cleaned_new_features)  # Standardize the features

    # Convert standardized features to tensors and move to device
    new_features_tensor = mx.array(scaled_new_features, dtype=mx.float32)

    # Use the trained model to make predictions on the new data
    # MLX computations are lazy and materialized when needed, hence no need for the evaluation mode or no_grad.
    predictions = model(new_features_tensor)

    # If the predictions are already materialized, we can print them directly
    print(predictions)

    # If you need to convert the predictions to a list or another Python-native format:
    predictions_np = np.array(predictions)

    # Output the predictions and the number of predictions made
    print(predictions_np)
    print(f"Number of predictions: {len(predictions_np)}")