import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Define paths and hyperparameters for the ML process
NON_NUMERIC_PLACEHOLDERS = ['#VALUE!', '-']  # Placeholder values for non-numeric data.
FOLDER_PATH = './data'                   # Path to the directory with CSV files for training.
PREDICTION_DATA_PATH = './new_data/raw_data-nflx.csv'  # Path to new data for making predictions.
# PREDICTION_DATA_PATH = './new_data/raw_data-avgo.csv'  # Path to new data for making predictions.
NUM_EPOCHS = 5000                        # Total number of training iterations over the dataset.
BATCH_SIZE = 100                         # Number of samples per batch to load.
HIDDEN_SIZE = 30                         # Number of units in hidden layers of the neural network.
OUTPUT_SIZE = 1                          # Number of units in the output layer (target prediction).
LEARNING_RATE = 0.0001                   # Step size at each iteration while moving toward a minimum of the loss function.
TRAIN_RATIO = 0.7                        # Proportion of dataset to include in the training split.
VAL_RATIO = 0.2                          # Proportion of dataset to include in the validation split.

# Determine the processing device based on availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class TennyDataset(TensorDataset):
    def __init__(self, folder_path, label_position, device='cpu', scaler=None, fit_scaler=False):
        super(TennyDataset, self).__init__()
        self.folder_path = folder_path
        self.label_position = label_position
        self.device = device
        self.scaler = scaler

        file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.train_files, self.val_files, self.test_files = self.split_data(file_names, TRAIN_RATIO, VAL_RATIO)

        # Call read_and_clean_data once and store the result
        self.data_df = self.read_and_clean_data(self.train_files)
        self.features, self.labels = self.prepare_features_labels(self.data_df)

        if fit_scaler:
            scaler.fit(self.features)

        # Convert the features and labels to tensors on the specified device
        self.features, self.labels = self.tensors_to_device(self.features, self.labels, device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def to_device(self, data, device):
        # Modify data in-place
        if isinstance(data, (list, tuple)):
            for x in data:
                x.to(device)
        else:
            data.to(device)
        return data

    def tensors_to_device(self, features, labels, device):
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return features_tensor.to(device), labels_tensor.to(device)

    def split_data(self, file_names, train_ratio, val_ratio):
        total_files = len(file_names)
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)

        train_files = file_names[:train_size]
        val_files = file_names[train_size:train_size + val_size]
        test_files = file_names[train_size + val_size:]

        return train_files, val_files, test_files

    def clean_data(self, df):
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()  # Work on this copy to ensure we're not modifying a slice

        # We're filling NaN values with the mean of each column. This is a simple imputation method, but it might not be the best strategy for all datasets. We might want to consider more sophisticated imputation methods, or make this a configurable option.

        # Replace non-numeric placeholders with NaN
        df_cleaned.replace(NON_NUMERIC_PLACEHOLDERS, pd.NA, inplace=True)

        # Ensure all data is numeric
        df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')

        # Fill NaN values in numerical columns with column mean
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype == 'float64' or df_cleaned[column].dtype == 'int64':
                df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)

        return df_cleaned

    def read_and_clean_data(self, files):
        # Read all files at once
        # Transposing each DataFrame after reading it could be a costly operation. If possible, we need to change the format of the data files to avoid the need for transposition
        data = pd.concat([pd.read_csv(os.path.join(self.folder_path, file), index_col=0).transpose() for file in files], ignore_index=True)
        data = self.clean_data(data)
        return data

    def prepare_features_labels(self, data_df):
        # Adjust for the fact that label_position is 1-indexed by subtracting 1 for 0-indexing
        label_idx = self.label_position - 1
        labels = data_df.iloc[:, label_idx]  # Extract labels from the specified position

        # In the prepare_features_labels method, dropping the label column from the features DataFrame creates a copy of the DataFrame, which could be memory-intensive for large datasets. Instead, we are using iloc to select only the columns you need for the features.

        # Select only the feature columns using iloc
        if label_idx == 0:
            features = data_df.iloc[:, 1:]  # If the label is the first column, select all columns after it
        else:
            # If the label is not the first column, select all columns before and after it
            features = pd.concat([data_df.iloc[:, :label_idx], data_df.iloc[:, label_idx + 1:]], axis=1)

        # Convert to numpy arrays and return
        return features.values, labels.values

    @staticmethod
    def create_datasets(folder_path, label_position, device, scaler, train_ratio, val_ratio, fit_scaler=False):
        # Create the train dataset
        train_dataset = TennyDataset(folder_path, label_position, device, scaler, fit_scaler)

        # Create the validation and test datasets
        val_dataset = TennyDataset(folder_path, label_position, device, scaler=scaler)
        test_dataset = TennyDataset(folder_path, label_position, device, scaler=scaler)

        return train_dataset, val_dataset, test_dataset


class TennyPredictionDataset(TennyDataset):
    def __init__(self, file_path, label_position, device='cpu', scaler=None, fit_scaler=False):
        super(TennyDataset, self).__init__()
        self.file_path = file_path
        self.folder_path = ''
        self.label_position = label_position
        self.device = device
        self.scaler = scaler

        # Call the parent class's read_and_clean_data method
        data_df = super().read_and_clean_data([file_path])
        self.features, self.labels = self.prepare_features_labels(data_df)

        if fit_scaler:
            scaler.fit(self.features)

        # Convert the features and labels to tensors on the specified device
        self.features, self.labels = self.tensors_to_device(self.features, self.labels, device)


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


def train(model, train_dataset, val_dataset, criterion, optimizer):
    # Instantiate the TennyDataset
    # Instantiate scaler

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize neural network model and move it to the appropriate computing device
    model = model.to(DEVICE)  # Move the model to the GPU if available
    best_val_loss = float('inf')  # Initialize best validation loss for early stopping

    # Early stopping with patience
    patience = 10
    no_improve = 0

    # Train the neural network
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Transfer data to the device
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
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Transfer data to the device
                outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
                val_loss += criterion(outputs, labels).item()  # Update total validation loss
        val_loss /= len(val_loader)  # Calculate the average loss over the validation set

        # Print training/validation statistics
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

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


def test(model, test_dataset, criterion=nn.MSELoss()):
    # Evaluate the model on the test dataset
    model.eval()  # Set the model to evaluation mode
    test_loss = 0  # Initialize variable to accumulate test loss
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():  # Disabling the gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Transfer data to the device
            outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
            test_loss += criterion(outputs, labels).item()  # Update total test loss
    test_loss /= len(test_loader)  # Calculate the average loss over the test set
    print(f"Average Test Loss: {test_loss:.4f}")


def predict(model):
    # Instantiate the dataset for prediction only
    prediction_dataset = TennyPredictionDataset(file_path=PREDICTION_DATA_PATH, label_position=1, scaler=scaler, device=DEVICE)

    # Process the file for prediction
    new_features_tensor = prediction_dataset.features

    # Use the trained model to make predictions on the new data
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disabling gradient calculation
        predictions = model(new_features_tensor)
        predictions_np = predictions.cpu().numpy()  # Transfer predictions back to CPU if they were on GPU

    print(predictions_np)
    print(f"Number of predictions: {len(predictions_np)}")


if __name__ == '__main__':
    # The dataset constructor will read, clean, and scale the data, and convert to tensors
    scaler = StandardScaler()
    train_dataset, val_dataset, test_dataset = TennyDataset.create_datasets(FOLDER_PATH, label_position=1, device=DEVICE, scaler=scaler, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, fit_scaler=True)
    input_size = train_dataset.features.shape[1]  # Determine input size from the training dataset
    model = Tenny(input_size=input_size, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    criterion = nn.MSELoss()  # Use Mean Squared Error Loss as the loss function for regression tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Use Adam optimizer as the optimization algorithm
    train(model, train_dataset, val_dataset, criterion, optimizer)
    test(model, test_dataset, criterion)
    predict(model)