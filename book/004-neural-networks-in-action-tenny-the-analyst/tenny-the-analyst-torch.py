import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Define paths and hyperparameters for the ML process
folder_path = './data'                   # Path to the directory with CSV files for training.
# prediction_data_path = './new_data/raw_data-nflx.csv'  # Path to new data for making predictions.
prediction_data_path = './new_data/raw_data-avgo.csv'  # Path to new data for making predictions.
num_epochs = 5000                        # Total number of training iterations over the dataset.
batch_size = 100                         # Number of samples per batch to load.
hidden_size = 30                         # Number of units in hidden layers of the neural network.
output_size = 1                          # Number of units in the output layer (target prediction).
learning_rate = 0.0001                   # Step size at each iteration while moving toward a minimum of the loss function.
train_ratio = 0.7                        # Proportion of dataset to include in the training split.
val_ratio = 0.2                          # Proportion of dataset to include in the validation split.

# Determine the processing device based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper function to convert and send data to the device
def to_device(data):
    if isinstance(data, (list, tuple)):
        return [to_device(x) for x in data]
    return data.to(device)


# Convert features and labels to tensors and send them to the device
# The first feature of the dataset is the label: Normalized Price
def tensors_to_device(features, labels):
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Labels need to be a 2D tensor
    return to_device(features_tensor), to_device(labels_tensor)


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

    # Define the forward pass through the network
    def forward(self, x):
        # Apply ReLU activations to linear layers and include dropout after the second layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
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

# Convert the scaled features and labels to tensors
train_features_tensor, train_labels_tensor = tensors_to_device(train_features_scaled, train_labels)
val_features_tensor, val_labels_tensor = tensors_to_device(val_features_scaled, val_labels)
test_features_tensor, test_labels_tensor = tensors_to_device(test_features_scaled, test_labels)

# Create tensor datasets from the tensors
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize neural network model and move it to the appropriate computing device
input_size = train_features_tensor.shape[1]  # Determine input size from the training dataset
model = Tenny(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
model = model.to(device)  # Move the model to the GPU if available
criterion = nn.MSELoss()  # Use Mean Squared Error Loss as the loss function for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Use Adam optimizer as the optimization algorithm
best_val_loss = float('inf')  # Initialize best validation loss for early stopping

# Early stopping with patience
patience = 10
no_improve = 0

# Train the neural network
for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set the model to training mode
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Transfer data to the device
        outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
        loss = criterion(outputs, labels)  # Calculate loss

        # Check for NaN in loss value to prevent invalid computations
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch + 1}")
            break

        # Gradient descent: clear previous gradients, compute gradients of all variables wrt loss, and make an optimization step
        optimizer.zero_grad()  # Zero the parameter gradients
        loss.backward()  # Backward pass: calculate gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0  # Initialize variable to accumulate validation loss
    with torch.no_grad():  # Disabling the gradient calculation to save memory and computations
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Transfer data to the device
            outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
            val_loss += criterion(outputs, labels).item()  # Update total validation loss
    val_loss /= len(val_loader)  # Calculate the average loss over the validation set

    # Print training/validation statistics
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

    # Check for improvement
    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        # Save model (commented out): torch.save(model.state_dict(), 'best_model.pth')
    else:
        # Stop training if there is no improvement observed
        no_improve += 1
        if no_improve == patience:
            print("No improvement in validation loss for {} epochs, stopping training.".format(patience))
            break
    model.train()  # Set the model back to training mode for the next epoch

# Evaluate the model on the test dataset
model.eval()  # Set the model to evaluation mode
test_loss = 0  # Initialize variable to accumulate test loss
with torch.no_grad():  # Disabling the gradient calculation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Transfer data to the device
        outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
        test_loss += criterion(outputs, labels).item()  # Update total test loss
test_loss /= len(test_loader)  # Calculate the average loss over the test set
print(f"Average Test Loss: {test_loss:.4f}")

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
new_features_tensor = torch.tensor(scaled_new_features, dtype=torch.float32).to(device)

# Use the trained model to make predictions on the new data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disabling the gradient calculation
    predictions = model(new_features_tensor)
    predictions_np = predictions.cpu().numpy()  # Transfer predictions back to CPU if they were on GPU

    # Optional: inverse transform the predictions if the target was originally scaled
    # predictions_original_scale = label_scaler.inverse_transform(predictions_np)

    # Output the predictions and the number of predictions made
    print(predictions_np)
    print(f"Number of predictions: {len(predictions_np)}")
