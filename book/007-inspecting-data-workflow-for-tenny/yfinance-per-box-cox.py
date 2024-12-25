import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

# Apply a Box-Cox transformation to the positive values only
# Adding a small constant because Box-Cox cannot handle zero or negative values
stock_data['ReturnOnEquity'] += 1e-9  # To handle 0 values, if any
stock_data['TrailingPE'] += 1e-9      # To handle 0 values, if any

roe_transformed, _ = stats.boxcox(stock_data['ReturnOnEquity'])
pe_transformed, _ = stats.boxcox(stock_data['TrailingPE'])

# Plot the transformed ROE
plt.figure(figsize=(10, 6))
sns.histplot(roe_transformed, kde=True, color='blue')
plt.title('Box-Cox Transformed Histogram of Return on Equity (ROE) for Selected Companies')
plt.xlabel('Box-Cox Transformed Return on Equity')
plt.ylabel('Frequency')
plt.show()

# Plot the transformed Trailing PE
plt.figure(figsize=(10, 6))
sns.histplot(pe_transformed, kde=True, color='blue')
plt.title('Box-Cox Transformed Histogram of Trailing PE for Selected Companies')
plt.xlabel('Box-Cox Transformed Trailing PE')
plt.ylabel('Frequency')
plt.show()
