import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import AlphaDropout

TRAIN_RATIO = 0.7                        # Proportion of dataset to include in the training split.
VAL_RATIO = 0.2                          # Proportion of dataset to include in the validation split.
NON_NUMERIC_PLACEHOLDERS = ['#VALUE!', '-']  # Placeholder values for non-numeric data.


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
        # self.dropout = nn.Dropout(0.5)  # To prevent overfitting
        self.dropout = AlphaDropout(0.5)  # Replace standard Dropout with AlphaDropout

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

