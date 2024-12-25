import yfinance as yf
import pandas as pd

# Define the ticker symbols
tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE"]

# Initialize an empty DataFrame for historical data
historical_data = pd.DataFrame()

for ticker in tickers:
    stock = yf.Ticker(ticker)

    # Fetch 10 years of historical data
    hist = stock.history(period="10y")

    # Resample the data to get quarterly data
    # We'll use 'quarterly mean' for demonstration, but you may choose a different method
    quarterly_data = hist.resample('Q').mean()

    # Add a column for the ticker symbol
    quarterly_data['Ticker'] = ticker

    # With the release of Pandas 2.0, several changes have been introduced, including the removal of the previously deprecated append method.
    # The recommended approach now is to use pd.concat for combining DataFrames.
    # Use pd.concat to append this data to the main DataFrame
    historical_data = pd.concat([historical_data, hist], ignore_index=True)

# Reset the index of the DataFrame
historical_data.reset_index(inplace=True)

# Display the DataFrame
print(historical_data)

#        index        Open        High  ...     Volume  Dividends  Stock Splits
# 0          0   16.827779   17.120530  ...  412610800        0.0           0.0
# 1          1   17.042887   17.094235  ...  317209200        0.0           0.0
# 2          2   16.870368   17.081713  ...  258529600        0.0           0.0
# 3          3   17.120530   17.122410  ...  279148800        0.0           0.0
# 4          4   16.902299   16.932671  ...  304976000        0.0           0.0
# ...      ...         ...         ...  ...        ...        ...           ...
# 25175  25175  596.090027  600.750000  ...    1893900        0.0           0.0
# 25176  25176  589.510010  590.440002  ...    2840200        0.0           0.0
# 25177  25177  574.580017  577.299988  ...    2478000        0.0           0.0
# 25178  25178  570.989990  572.909973  ...    2092100        0.0           0.0
# 25179  25179  563.500000  569.520020  ...    1922900        0.0           0.0
#
# [25180 rows x 8 columns]