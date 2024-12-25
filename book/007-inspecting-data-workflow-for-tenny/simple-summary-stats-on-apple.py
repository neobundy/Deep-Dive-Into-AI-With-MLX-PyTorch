import pandas as pd

# Replace with your actual file name
file_name = './data/raw_data-aapl.csv'

# Reading the data file into a DataFrame and transposing it
df = pd.read_csv(file_name).T

# Note that we can also use the following code to transpose the DataFrame:
# df = pd.read_csv('file_name').transpose()

# Resetting the header
df.columns = df.iloc[0]
df = df.drop(df.index[0])

# Convert all columns to float for statistical analysis
df = df.astype(float)

# Compute summary statistics for all columns (now features)
summary_stats = df.describe()

print("Summary Statistics for All Features")
print(summary_stats)

# If you want to compute summary statistics for a specific column (feature), for example 'Normalized Price'
normalized_price_stats = df['Normalized Price'].describe()

print("Summary Statistics for the 'Normalized Price' Feature")
print(normalized_price_stats)
