import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define the ticker symbols
tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", "NFLX", "AVGO"]

# Initialize an empty DataFrame for stock data
stock_data = pd.DataFrame()

for ticker in tickers:
    stock = yf.Ticker(ticker)

    # Get current info
    info = stock.info
    selected_info = {
        'Ticker': ticker,
        'ReturnOnEquity': info.get('returnOnEquity'),
        'TrailingPE': info.get('trailingPE')
    }

    # Create a DataFrame from the selected info
    ticker_df = pd.DataFrame([selected_info])

    # Concatenate the new DataFrame with the existing one
    stock_data = pd.concat([stock_data, ticker_df], ignore_index=True)

# Remove rows with NaN
stock_data.dropna(inplace=True)

# Apply log transformation to the positive values only
# Adding a small constant to shift any zero values in the data
stock_data['ReturnOnEquity_Log'] = np.log(stock_data['ReturnOnEquity'] + 1e-9)
stock_data['TrailingPE_Log'] = np.log(stock_data['TrailingPE'] + 1e-9)

# Plot the transformed ROE
plt.figure(figsize=(10, 6))
sns.histplot(stock_data['ReturnOnEquity_Log'], kde=True, color='blue')
plt.title('Log Transformed Histogram of Return on Equity (ROE) for Selected Companies')
plt.xlabel('Log Transformed Return on Equity')
plt.ylabel('Frequency')
plt.show()

# Plot the transformed Trailing PE
plt.figure(figsize=(10, 6))
sns.histplot(stock_data['TrailingPE_Log'], kde=True, color='blue')
plt.title('Log Transformed Histogram of Trailing PE for Selected Companies')
plt.xlabel('Log Transformed Trailing PE')
plt.ylabel('Frequency')
plt.show()