import pandas as pd
import os
import random

# Folder containing the CSV files
folder_path = './data'

# List all CSV files in the folder and shuffle
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
random.shuffle(file_names)  # Randomly shuffle the list

# Split the companies
train_files = file_names[:7]
val_files = file_names[7:9]
test_files = file_names[9:]

# Function to read data from files
def read_data(files):
    data = pd.DataFrame()
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = pd.concat([data, pd.read_csv(file_path)], ignore_index=True)
    return data

# Create datasets
train_data = read_data(train_files)
val_data = read_data(val_files)
test_data = read_data(test_files)

# Print the shape of each dataset
print("Train:", train_data.shape, "Validation:", val_data.shape, "Test:", test_data.shape)

# Access the first few rows of each dataset
print("Training Data:")
print(train_data.head())

print("\nValidation Data:")
print(val_data.head())

print("\nTest Data:")
print(test_data.head())
