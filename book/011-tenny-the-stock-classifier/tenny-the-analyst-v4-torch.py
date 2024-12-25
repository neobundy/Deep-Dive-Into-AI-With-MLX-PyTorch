import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tenny import TennyDataset, TennyPredictionDataset, Tenny

# Define paths and hyperparameters for the ML process
TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", 'NFLX', 'AVGO']
GROWTH_STOCK = 'Growth'
STALWART_STOCK = 'Stalwart'
OTHER_STOCK = 'Other'
CLASS_INDICES = {GROWTH_STOCK: 0, STALWART_STOCK: 1, OTHER_STOCK: 2}
CLASS_LABELS = {0: 'Growth', 1: 'Stalwart', 2: 'Other'}
NUM_CLASSES = 3 # Growth, Stalwart, Other
NON_NUMERIC_PLACEHOLDERS = ['#VALUE!', '-']  # Placeholder values for non-numeric data.
FOLDER_PATH = './enhanced-data-with-labels' # Path to the directory with CSV files for training.
CSV_FILE_PREFIX = 'labeled-enhanced-raw-data' # Prefix for CSV files in the folder.
NUM_EPOCHS = 1000                        # Total number of training iterations over the dataset.
BATCH_SIZE = 20                         # Number of samples per batch to load.
HIDDEN_SIZE = 20                         # Number of units in hidden layers of the neural network.
LEARNING_RATE = 0.0001                   # Step size at each iteration while moving toward a minimum of the loss function.
TRAIN_RATIO = 0.7                        # Proportion of dataset to include in the training split.
VAL_RATIO = 0.2                          # Proportion of dataset to include in the validation split.

# Early stopping with patience
PATIENCE = 20  # Number of epochs to wait for improvement before early stopping

L1_LAMBDA = 0.001
L2_LAMBDA = 0.001

# Model to Use: Complex vs. Simple
USE_COMPLEX_MODEL = True

# Determine the processing device based on availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class TennyClassifierDataset(TennyDataset):
    def __init__(self, folder_path, label_position, device='cpu', scaler=None, fit_scaler=False):
        super(TennyClassifierDataset, self).__init__(folder_path, label_position, device='cpu', scaler=scaler, fit_scaler=fit_scaler)

    def clean_data(self, df):
        df_cleaned = df.copy()

        # Replace non-numeric placeholders with NaN for all columns except 'Label'
        for col in df_cleaned.columns:
            if col != 'Label':
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        # Fill NaN values in numerical columns with column mean, again excluding 'Label'
        for column in df_cleaned.columns:
            if column != 'Label' and (df_cleaned[column].dtype == 'float64' or df_cleaned[column].dtype == 'int64'):
                df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)

        return df_cleaned

    def prepare_features_labels(self, df):
        # Assuming the last column is the relevant label
        labels = df.iloc[:, -1]  # Only the last label is used
        features = df.iloc[:, :-1]  # All columns except the last one

        # Convert labels to class indices
        labels = labels.map(self.one_hot_encode).values

        return features.values, labels

    def tensors_to_device(self, features, labels, device):
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)  # Note the dtype change here
        return features_tensor.to(device), labels_tensor.to(device)

    def read_and_clean_data(self, files):
        data = super().read_and_clean_data(files)

        # Select only the most recent value for constant features
        # After transposing, each of these features is a single-row dataframe or series.
        # We should take the value directly.
        data['Beta'] = data['Beta'].values[0]
        data['Industry Beta Average'] = data['Industry Beta Average'].values[0]
        data['Industry PE Average'] = data['Industry PE Average'].values[0]

        return data

    @staticmethod
    def create_datasets(folder_path, label_position, device, scaler, train_ratio, val_ratio, fit_scaler=False):
        # Create the train dataset
        train_dataset = TennyClassifierDataset(folder_path, label_position, device, scaler, fit_scaler)

        # Create the validation and test datasets
        val_dataset = TennyClassifierDataset(folder_path, label_position, device, scaler=scaler)
        test_dataset = TennyClassifierDataset(folder_path, label_position, device, scaler=scaler)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def one_hot_encode(stock_category):
        return CLASS_INDICES[stock_category]


class TennyClassifierPredictionDataset(TennyPredictionDataset):

    def read_and_clean_data(self, files):
        data = super().read_and_clean_data(files)
        aggregated_data = self.aggregate_data(data)
        return aggregated_data

    def aggregate_data(self, data_df):
        aggregated_features = []
        for column in data_df.columns:
            if column in ['Beta', 'Industry Beta Average', 'Industry PE Average']:
                aggregated_features.append(data_df[column].iloc[-1])
            else:
                aggregated_features.extend(data_df[column])
        return np.array(aggregated_features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Define the new neural network architecture
class TennyClassifier(Tenny):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TennyClassifier, self).__init__(input_size, hidden_size, num_classes)
        # The final fully connected layer's output size is set based on `num_classes`
        self.fc5 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)  # Applying softmax to get probabilities


class TennyClassifierSimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TennyClassifierSimple, self).__init__()

        # Adjust the number of layers and neurons per layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Reduced number of neurons
        self.dropout = nn.Dropout(0.3)  # Adjusted dropout rate
        # The final fully connected layer's output size is set based on `num_classes`
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # Directly connecting to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)  # Applying softmax to get probabilities

def evaluate_metrics(outputs, labels):
    _, predicted_classes = torch.max(outputs, 1)
    accuracy = accuracy_score(labels.cpu(), predicted_classes.cpu())
    precision = precision_score(labels.cpu(), predicted_classes.cpu(), average='macro', zero_division=0)
    recall = recall_score(labels.cpu(), predicted_classes.cpu(), average='macro', zero_division=0)
    f1 = f1_score(labels.cpu(), predicted_classes.cpu(), average='macro')
    return accuracy, precision, recall, f1


def train(model, train_dataset, val_dataset, criterion, optimizer):
    # Create weighted sampler for the training dataset
    labels = train_dataset.labels

    # Calculate the class weights
    class_counts = torch.bincount(labels)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # Create the DataLoader with the sampler
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize neural network model and move it to the appropriate computing device
    model = model.to(DEVICE)  # Move the model to the GPU if available
    best_val_loss = float('inf')  # Initialize best validation loss for early stopping

    # Early stopping with patience
    no_improve = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=PATIENCE/2, verbose=True)

    # Train the neural network
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Transfer data to the device
            outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass: calculate gradient of the loss with respect to model parameters
            optimizer.step()  # Perform a single optimization step

        # Validation phase
        model.eval()
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Transfer data to the device
                outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
                val_loss += criterion(outputs, labels).item()
                acc, prec, rec, f1 = evaluate_metrics(outputs, labels)
                val_accuracy += acc
                val_precision += prec
                val_recall += rec
                val_f1 += f1
        # Average the metrics over the validation set
        val_accuracy /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Prec: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        # Check for improvement
        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # Save model (commented out): torch.save(model.state_dict(), 'best_model.pth')
        else:
            # Stop training if there is no improvement observed
            no_improve += 1
            if no_improve == PATIENCE:
                print("No improvement in validation loss for {} epochs, stopping training.".format(PATIENCE))
                break
        # Step the scheduler
        scheduler.step(val_loss)
        model.train()  # Set the model back to training mode for the next epoch


def test(model, test_dataset, criterion):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            acc, prec, rec, f1 = evaluate_metrics(outputs, labels)
            test_accuracy += acc
            test_precision += prec
            test_recall += rec
            test_f1 += f1
    # Average the metrics over the test set
    test_accuracy /= len(test_loader)
    test_precision /= len(test_loader)
    test_recall /= len(test_loader)
    test_f1 /= len(test_loader)
    test_loss /= len(test_loader)

    print(f"Average Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test Prec: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")


# Prediction function
def predict_single(model, prediction_dataset):
    model.eval()
    with torch.no_grad():
        # Assuming the dataset returns a tuple (features, label)
        # and we are using the first item in the dataset for prediction
        features, _ = prediction_dataset[0]
        features_tensor = features.unsqueeze(0).to(DEVICE)  # Add batch dimension and send to device

        # Get prediction from model
        prediction = model(features_tensor)

        # Process the prediction
        predicted_index = torch.argmax(prediction, dim=1).item()
        return CLASS_LABELS[predicted_index]


if __name__ == '__main__':
    scaler = StandardScaler()

    predictions = []

    for excluded_ticker in TICKERS:
        print(f"Training model on all companies except {excluded_ticker}. Predicting for {excluded_ticker}.")

        # Construct the filename for the excluded ticker
        prediction_file_name = f"{FOLDER_PATH}/{CSV_FILE_PREFIX}-{excluded_ticker.lower()}.csv"

        # Filter out the file for the excluded ticker
        train_val_files = [f for f in os.listdir(FOLDER_PATH) if f != prediction_file_name]

        # Determine the label position based on the number of columns
        num_columns = len(pd.read_csv(prediction_file_name).transpose().columns)
        label_position = num_columns
        # Create train, validation, and prediction datasets
        train_dataset, val_dataset, test_dataset = TennyClassifierDataset.create_datasets(
            FOLDER_PATH, label_position=label_position, device=DEVICE, scaler=scaler,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, fit_scaler=True)
        prediction_dataset = TennyClassifierPredictionDataset(
            file_path=prediction_file_name,
            label_position=label_position, device=DEVICE, scaler=scaler, fit_scaler=False
        )

        # Define the model
        input_size = train_dataset.features.shape[1]

        # experiment with different models
        if USE_COMPLEX_MODEL:
            model = TennyClassifier(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
        else:
            model = TennyClassifierSimple(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)

        criterion = nn.CrossEntropyLoss()

        # PyTorch's optimizers, such as `Adam`, have built-in support for L2 regularization via the `weight_decay` parameter.
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)

        train(model, train_dataset, val_dataset, criterion, optimizer)
        test(model, test_dataset, criterion)
        # Make predictions for the excluded ticker
        # predict(model, prediction_dataset)
        prediction = predict_single(model, prediction_dataset)

        predictions.append(f"Predictions for {excluded_ticker}: {prediction}")


print('\n'.join(predictions))

