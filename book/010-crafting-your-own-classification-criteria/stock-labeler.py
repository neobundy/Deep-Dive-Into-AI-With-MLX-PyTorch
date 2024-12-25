import pandas as pd
import os
import numpy as np
from scipy import stats

# Constants
NON_NUMERIC_PLACEHOLDERS = ['#VALUE!', '-']
GROWTH_STOCK = 'Growth'
STALWART_STOCK = 'Stalwart'
OTHER_STOCK = 'Other'
MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS = 4
BETA_SCALE_THRESHOLD = 0.8


# Utility Functions
def print_reason(ticker, label, reason):
    print(f"{ticker} is disqualified as {label}: {reason}")


def read_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.replace(NON_NUMERIC_PLACEHOLDERS, np.nan, inplace=True)
        df = df.transpose()  # Transposing the data
        df.columns = df.iloc[0]  # Set the first row as column headers
        df = df[1:]  # Exclude the original header row
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def is_dividend_initiated(dividend_yields):
    """Check if dividends are initiated for the first time."""
    return float(dividend_yields[-1]) > 0 and all(d == 0 or pd.isna(d) for d in dividend_yields[:-1])


def check_cagr_trend(recent_cagr_quarters):
    """Check for the trend in CAGR."""
    consecutive_decreasing_quarters = 0
    for i in range(1, len(recent_cagr_quarters)):
        if float(recent_cagr_quarters[i]) < float(recent_cagr_quarters[i - 1]):
            consecutive_decreasing_quarters += 1
        else:
            break
    return consecutive_decreasing_quarters

# Core Functions
def classify_stock(ticker, df):
    # Extracting relevant indicators
    industry_pe_avg = df['Industry PE Average'].iloc[-1]  # Industry average P/E Ratio
    industry_beta_avg = df['Industry Beta Average'].iloc[-1]  # Industry average Beta
    cagr_revenue_quarters = df['Total Revenues / CAGR 5Y'].dropna().values
    dividend_yields = df['Dividend Yield (LTM)'].values
    beta = df['Beta'].dropna().values[-1]
    pe_ratio = df['Price / Earnings - P/E (LTM)'].iloc[-1]

    is_growth_stock = True
    is_cagr_decreasing = False

    # Checking for consecutive quarters of decreasing CAGR at the end of the series
    # Note that we are only checking the most recent quarters: the previous quarters are not relevant
    recent_cagr_quarters = cagr_revenue_quarters[-(MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS + 1):]
    consecutive_decreasing_quarters = check_cagr_trend(recent_cagr_quarters)
    if consecutive_decreasing_quarters > 1:
        is_cagr_decreasing = True

    if consecutive_decreasing_quarters >= MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS:
        is_growth_stock = False  # Disqualify if recent CAGR is decreasing for the specified number of consecutive quarters
        print_reason(ticker, GROWTH_STOCK, f"Recent CAGR is decreasing for {MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS} consecutive quarters")

    # Check if it's the first time dividends are initiated (all previous are nil) during a declining CAGR
    if is_dividend_initiated(dividend_yields) and is_cagr_decreasing:
        is_growth_stock = False
        print_reason(ticker, GROWTH_STOCK, f"Distributing dividends for the first time during a period of declining CAGR")

    # Checking Beta criteria
    # Beta lower than (BETA_SCALE_THRESHOLD * industry average) is only acceptable if CAGR is increasing
    # If the industry average is 1.5 and the threshold is 0.8, then the threshold is 1.2. This means that a Beta of 1.1 is acceptable only if CAGR is increasing.
    if float(beta) < (BETA_SCALE_THRESHOLD * industry_beta_avg) and float(cagr_revenue_quarters[-1]) < 0:
        is_growth_stock = False
        print_reason(ticker, GROWTH_STOCK, f"Below threshold Beta ({BETA_SCALE_THRESHOLD * industry_beta_avg}) during a declining CAGR")

    # Checking P/E Ratio criteria
    # P/E Ratio lower than industry average is only acceptable if CAGR is increasing
    # Negative P/E Ratio is not acceptable even if CAGR is increasing: this is a sign of a loss-making company, we need closer inspection
    if pd.isna(pe_ratio) or pd.isna(industry_pe_avg):
        print_reason(ticker, f"either {GROWTH_STOCK} nor {STALWART_STOCK}", f"Negative P/E Ratio or industry average P/E Ratio is not available. Needs more closer inspection.")
        print(f"{ticker} is classified as {OTHER_STOCK}" )
        return OTHER_STOCK
    elif float(pe_ratio) < industry_pe_avg and is_cagr_decreasing:
        print_reason(ticker,GROWTH_STOCK, f"P/E Ratio is below industry average ({industry_pe_avg}) during a period of declining CAGR")
        is_growth_stock = False

    if is_growth_stock:
        print(f"{ticker} is classified as {GROWTH_STOCK}" )
        return GROWTH_STOCK
    else:
        print(f"{ticker} is classified as {STALWART_STOCK}" )
        return STALWART_STOCK


def calculate_industry_averages(tickers, enhanced_data_folder):
    beta_values, pe_values = [], []
    for ticker in tickers:
        file_path = os.path.join(enhanced_data_folder, f"beta-enhanced-raw-data-{ticker.lower()}.csv")
        df = read_and_clean_data(file_path)
        if df is not None:
            # Get the latest Beta value
            beta_values.append(float(df['Beta'].dropna().iloc[-1]))

            # Get the latest P/E ratio
            latest_pe = pd.to_numeric(df['Price / Earnings - P/E (LTM)'].dropna().iloc[-1], errors='coerce')

            if not pd.isna(latest_pe):
                pe_values.append(latest_pe)

    # Calculate and return the industry averages
    industry_beta_avg = round(np.mean(beta_values), 2) if beta_values else 0

    # We could have outliers in the P/E ratio data, so we need to normalize the data before calculating the average
    # Trimmed mean approach
    trimmed_mean_pe = stats.trim_mean(pe_values, 0.1) if pe_values else 0  # Trim 10% from each end
    # Median approach
    median_pe = np.median(pe_values) if pe_values else 0

    # you can choose either the trimmed mean or the median
    # industry_pe_avg = round(trimmed_mean_pe, 2)
    industry_pe_avg = round(median_pe, 2)
    return industry_beta_avg, industry_pe_avg


def label_growth_stocks(tickers, enhanced_data_folder, enhanced_data_with_labels_folder, industry_beta_avg, industry_pe_avg):
    for ticker in tickers:
        file_path = os.path.join(enhanced_data_folder, f"beta-enhanced-raw-data-{ticker.lower()}.csv")
        df = read_and_clean_data(file_path)
        if df is not None:
            df['Industry Beta Average'] = industry_beta_avg
            df['Industry PE Average'] = industry_pe_avg
            df['Label'] = classify_stock(ticker, df)

            # Transposing the data back to its original format
            df = df.transpose()

            # Saving the updated DataFrame
            new_file_name = f"labeled-enhanced-raw-data-{ticker.lower()}.csv"
            new_file_path = os.path.join(enhanced_data_with_labels_folder, new_file_name)
            df.to_csv(new_file_path, index=True)


# Main Execution
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", 'NFLX', 'AVGO']
    enhanced_data_folder = "./enhanced-data"
    enhanced_data_with_labels_folder = "./enhanced-data-with-labels"

    industry_beta_avg, industry_pe_avg = calculate_industry_averages(tickers, enhanced_data_folder)
    label_growth_stocks(tickers, enhanced_data_folder, enhanced_data_with_labels_folder, industry_beta_avg, industry_pe_avg)
