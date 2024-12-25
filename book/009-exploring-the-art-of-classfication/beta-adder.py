import yfinance as yf
import pandas as pd
import numpy as np
import os

# List of tickers for which you want to pull Beta values
tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", 'NFLX', 'AVGO']
data_folder = "./data"
enhanced_data_folder = "./enhanced-data"

# Initialize a dictionary to store Beta values
beta_values = {}

# Retrieve and store Beta for each ticker
for ticker in tickers:
    stock = yf.Ticker(ticker)
    beta_values[ticker] = stock.info.get('beta')

# Update each company's dataframe with the Beta value and save as a new CSV
for ticker in tickers:
    file_name = f"enhanced-raw-data-{ticker.lower()}.csv"
    file_path = os.path.join(data_folder, file_name)

    # Read the company's dataframe
    df = pd.read_csv(file_path)

    # Create a new row for Beta
    beta_row = pd.DataFrame([['Beta'] + [np.nan] * (len(df.columns) - 2) + [beta_values[ticker]]], columns=df.columns)

    # Append the new row to the DataFrame using pd.concat()
    df = pd.concat([df, beta_row], ignore_index=True)

    print(df.tail())

    # Save the updated dataframe as a new CSV file
    new_file_name = f"beta-{file_name}"
    new_file_path = os.path.join(enhanced_data_folder, new_file_name)
    df.to_csv(new_file_path, index=False)


# Note: As of 2024-01-15, Beta values are as printed above
# This code assumes that the CSV files exist in the specified folder and follows the naming pattern mentioned.

# As of 2024-01-15
# AAPL: Beta = 1.29
# MSFT: Beta = 0.876
# AMZN: Beta = 1.163
# TSLA: Beta = 2.316
# GOOGL: Beta = 1.054
# META: Beta = 1.221
# NVDA: Beta = 1.642
# INTC: Beta = 0.995
# AMD: Beta = 1.695
# ADBE: Beta = 1.33
# NFLX: Beta = 1.283
# AVGO: Beta = 1.241