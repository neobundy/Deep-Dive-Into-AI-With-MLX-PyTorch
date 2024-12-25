import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Folder containing the CSV files
folder_path = './data'

# Function to clean data
def clean_data(df):
    # Replace non-numeric placeholders with NaN
    df.replace(['#VALUE!', '-'], [pd.NA, pd.NA], inplace=True)

    # Fill NaNs in numerical columns with column mean
    for column in df.select_dtypes(include=['float64', 'int64']):
        df[column].fillna(df[column].mean(), inplace=True)

    return df

# List all CSV files in the folder and shuffle
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
random.shuffle(file_names)  # Randomly shuffle the list

# Split the companies
train_files = file_names[:7]
val_files = file_names[7:9]
test_files = file_names[9:]

# Function to read and clean data from files
def read_and_clean_data(files):
    data = pd.DataFrame()
    for file in files:
        file_path = os.path.join(folder_path, file)
        temp_df = pd.read_csv(file_path)
        temp_df = clean_data(temp_df)  # Clean the data
        data = pd.concat([data, temp_df], ignore_index=True)
    return data

# Create datasets
train_data = read_and_clean_data(train_files)
val_data = read_and_clean_data(val_files)
test_data = read_and_clean_data(test_files)

for df in [train_data, val_data, test_data]:
    columns_to_drop = [col for col in ['4Q FY2023', '1Q FY2024'] if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)

# Print the shape of each dataset
print("Train:", train_data.shape, "Validation:", val_data.shape, "Test:", test_data.shape)

# Access the first few rows of each dataset
print("Training Data:")
print(train_data)

print("\nValidation Data:")
print(val_data)

print("\nTest Data:")
print(test_data)

class Tenny(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Tenny, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Example: input_size = number of features, output_size = 1 for price prediction
model = Tenny(input_size=10, hidden_size=20, output_size=1)
print(model)