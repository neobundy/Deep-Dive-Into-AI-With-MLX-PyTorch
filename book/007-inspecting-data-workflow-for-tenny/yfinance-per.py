import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    # Remove rows with NaN
    # stock_data.dropna(inplace=True)
    # Concatenate the new DataFrame with the existing one
    stock_data = pd.concat([stock_data, ticker_df], ignore_index=True)

# Display the DataFrame
print(stock_data)

# Setting the aesthetic style of the plots
sns.set(style="darkgrid")

# Plotting the histogram
plt.figure(figsize=(10, 6))
sns.histplot(stock_data['ReturnOnEquity'], kde=True, color='blue')

# Adding title and labels
plt.title('Histogram of Return on Equity (ROE) for Selected Companies')
plt.xlabel('Return on Equity (%)')
plt.ylabel('Frequency')

# Show the plot
plt.show()

plt.title('Histogram of Trailing PE for Selected Companies')
plt.xlabel('Trailing PE')
plt.ylabel('Frequency')

sns.histplot(stock_data['TrailingPE'], kde=True, color='blue')
plt.show()