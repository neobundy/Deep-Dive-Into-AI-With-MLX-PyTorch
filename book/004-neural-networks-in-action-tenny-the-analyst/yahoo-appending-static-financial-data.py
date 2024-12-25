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

    # Get current info
    info = stock.info
    selected_info = {
        'CurrentPrice': info.get('currentPrice'),
        'MarketCap': info.get('marketCap'),
        'BookValue': info.get('bookValue'),
        'ProfitMargins': info.get('profitMargins'),
        'EarningsGrowth': info.get('earningsGrowth'),
        'RevenueGrowth': info.get('revenueGrowth'),
        'ReturnOnEquity': info.get('returnOnEquity'),
        'ForwardEPS': info.get('forwardEps'),
        'TrailingEPS': info.get('trailingEps'),
        'ForwardPE': info.get('forwardPE'),
        'TrailingPE': info.get('trailingPE'),
        'FreeCashflow': info.get('freeCashflow')
    }

    # Repeat the info data for each date in the historical data
    for key, value in selected_info.items():
        hist[key] = value

    # Add a column for the ticker symbol
    hist['Ticker'] = ticker

    # Use pd.concat to append this data to the main DataFrame
    historical_data = pd.concat([historical_data, hist], ignore_index=True)

# Reset the index of the DataFrame
historical_data.reset_index(inplace=True, drop=True)

# Display the DataFrame
print(historical_data)

#              Open        High         Low  ...  TrailingPE  FreeCashflow  Ticker
# 0       16.827785   17.120536   16.707239  ...   29.604574   82179997696    AAPL
# 1       17.042881   17.094229   16.842495  ...   29.604574   82179997696    AAPL
# 2       16.870370   17.081715   16.866614  ...   29.604574   82179997696    AAPL
# 3       17.120538   17.122418   16.762034  ...   29.604574   82179997696    AAPL
# 4       16.902305   16.932677   16.629279  ...   29.604574   82179997696    AAPL
# ...           ...         ...         ...  ...         ...           ...     ...
# 25175  596.090027  600.750000  592.940002  ...   47.847454          None    ADBE
# 25176  589.510010  590.440002  576.760010  ...   47.847454          None    ADBE
# 25177  574.580017  577.299988  570.190002  ...   47.847454          None    ADBE
# 25178  570.989990  572.909973  566.659973  ...   47.847454          None    ADBE
# 25179  563.500000  569.520020  563.340027  ...   47.847454          None    ADBE
#
# [25180 rows x 20 columns]