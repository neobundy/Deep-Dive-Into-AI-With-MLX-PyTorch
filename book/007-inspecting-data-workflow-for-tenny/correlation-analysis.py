import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = './data/raw_data-aapl.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to ensure it's loaded correctly
data.head()

# Convert the 'Fiscal Quarters' row to header and transpose the DataFrame
data = data.set_index('Fiscal Quarters').transpose()

# Convert all columns to numeric, errors='coerce' will replace non-numeric values with NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Drop any non-numeric columns that could not be converted
data = data.dropna(axis=1, how='all')

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix for Financial Metrics')
plt.xticks(rotation=45)  # Rotate the x labels for better readability
plt.yticks(rotation=0)   # Keep the y labels horizontal
plt.show()

