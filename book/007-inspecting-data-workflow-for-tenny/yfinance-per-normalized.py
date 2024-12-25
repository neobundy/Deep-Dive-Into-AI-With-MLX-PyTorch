import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# Normalize the 'ReturnOnEquity' and 'TrailingPE' columns
scaler = StandardScaler()
stock_data[['ReturnOnEquity', 'TrailingPE']] = scaler.fit_transform(
    stock_data[['ReturnOnEquity', 'TrailingPE']])

# Display the DataFrame
print(stock_data)

# Setting the aesthetic style of the plots
sns.set(style="darkgrid")

# Plotting the histogram for normalized Return on Equity (ROE)
plt.figure(figsize=(10, 6))
sns.histplot(stock_data['ReturnOnEquity'], kde=True, color='blue')
plt.title('Normalized Histogram of Return on Equity (ROE) for Selected Companies')
plt.xlabel('Normalized Return on Equity')
plt.ylabel('Frequency')
plt.show()

# Plotting the histogram for normalized Trailing PE
plt.figure(figsize=(10, 6))
sns.histplot(stock_data['TrailingPE'], kde=True, color='blue')
plt.title('Normalized Histogram of Trailing PE for Selected Companies')
plt.xlabel('Normalized Trailing PE')
plt.ylabel('Frequency')
plt.show()
