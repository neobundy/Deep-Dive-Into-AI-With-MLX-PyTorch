def is_growth_stock(data):
    """
    Determines if a stock is a growth stock based on specific criteria.

    :param data: A dictionary containing a stock's quarterly financial data.
    :return: Boolean indicating whether the stock is a growth stock.
    """
    # Extracting relevant indicators
    cagr_revenue_quarters = data['total_revenues_cagr_5y']  # List of CAGR of Total Revenues for each quarter
    dividend_yield = data['dividend_yield_ltm'][-1]  # Latest Dividend Yield
    beta = data['beta'][-1]  # Latest Beta value
    pe_ratio = data['price_earnings_pe_ltm'][-1]  # Latest P/E Ratio
    industry_pe_avg = data['industry_pe']  # Industry average P/E Ratio

    # Checking for consecutive quarters of decreasing CAGR
    consecutive_decreasing_quarters = 0
    for i in range(1, len(cagr_revenue_quarters)):
        if cagr_revenue_quarters[i] < cagr_revenue_quarters[i - 1]:
            consecutive_decreasing_quarters += 1
        else:
            consecutive_decreasing_quarters = 0  # Reset count if CAGR does not decrease

    if consecutive_decreasing_quarters >= 4:
        return False  # Disqualify if CAGR decreases for four or more consecutive quarters

    # Checking if dividends are initiated during falling CAGR
    if dividend_yield > 0 and cagr_revenue_quarters[-1] < 0:
        return False

    # Checking Beta criteria
    if beta >= 1 and cagr_revenue_quarters[-1] < 0:
        return False

    # Checking P/E Ratio criteria
    if pe_ratio < industry_pe_avg and cagr_revenue_quarters[-1] < 0:
        return False

    return True  # Stock meets all criteria for being a growth stock

# Example Usage
growth_stock_candidate_1 = {
    'total_revenues_cagr_5y': [0.08, 0.085, 0.09, 0.07, 0.06],  # Example CAGR of revenue for recent quarters
    'dividend_yield_ltm': [0.01],  # Example Dividend Yield
    'beta': [1.29],  # Example Beta value
    'price_earnings_pe_ltm': [16.4, 17.2, 18.6, 12.8, 16.9],  # Example P/E ratios for recent quarters
    'industry_pe': 20,  # Example industry average P/E ratio
}

growth_stock_candidate_2 = {
    'total_revenues_cagr_5y': [0.08, 0.07, 0.06, 0.05, 0.04],  # Four consecutive quarters of falling CAGR
    'dividend_yield_ltm': [0.01],  # Example Dividend Yield
    'beta': [0.8],  # Example Beta value
    'price_earnings_pe_ltm': [16.4, 17.2, 18.6, 12.8, 16.9],  # Example P/E ratios for recent quarters
    'industry_pe': 20,  # Example industry average P/E ratio
}

if is_growth_stock(growth_stock_candidate_1):
    print("This is a growth stock.")
else:
    print("This is not a growth stock.")

if is_growth_stock(growth_stock_candidate_2):
    print("This is a growth stock.")
else:
    print("This is not a growth stock.")

