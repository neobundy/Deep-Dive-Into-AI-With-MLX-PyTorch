import pandas as pd

# Path to the CSV file containing the data
data_file = './beta-enhanced-raw-data-aapl.csv'

# Reading the CSV file into a pandas DataFrame
df = pd.read_csv(data_file)

# Displaying the first few rows of the DataFrame to understand the original structure
# This is the 'wide format', where each metric is in its own row and each quarter is a column
print("----- Original Data (Wide Format) -----")

print(df.head())

# ----- Original Data (Wide Format) -----
#                 Fiscal Quarters  2Q FY2018  ...  3Q FY2023  4Q FY2023
# 0              Normalized Price    42.6400  ...   191.9580   178.6400
# 1  Price / Earnings - P/E (LTM)    16.4000  ...    32.1000    29.0000
# 2               Net EPS - Basic     2.6000  ...     5.9800     6.1600
# 3            Return On Equity %     0.4086  ...     1.6009     1.7195
# 4      Total Revenues / CAGR 5Y     0.0791  ...     0.0851     0.0761

# Transposing the DataFrame
# In the transposed DataFrame, each row now corresponds to a fiscal quarter
# and each column corresponds to a different financial metric
# Note: This transposed version is still in a 'wide format' but now aligns with typical machine learning data structure
print("----- Transposed Data (Still Wide Format) -----")
df_transposed = df.transpose()
print(df_transposed.head())

# 10 columns = 10 different metrics or features
# ----- Transposed Data (Still Wide Format) -----
#                                 0  ...     9
# Fiscal Quarters  Normalized Price  ...  Beta
# 2Q FY2018                   42.64  ...   NaN
# 3Q FY2018                  47.816  ...   NaN
# 4Q FY2018                    55.8  ...   NaN
# 1Q FY2019                  39.168  ...   NaN
#
# [5 rows x 10 columns]

# Converting the original DataFrame to a 'long format' using the melt function
# 'id_vars' is set to ['Fiscal Quarters'] to keep the quarter names as a separate column
# 'var_name' is set to 'Indicators' - this will be the name of the new column created from the header of the original DataFrame
# 'value_name' is set to 'Values' - this will be the name of the new column containing the values from the original DataFrame
# Each row in this long format represents a single observation for a specific metric in a specific quarter
print("----- The Long Format of the Original Data -----")
df_long = pd.melt(df, id_vars=['Fiscal Quarters'], var_name='Indicators', value_name='Values')
print(df_long)

# In the long format, the DataFrame expands to 230 rows. This expansion results from
# combining each of the 23 quarters with each of the 10 different financial indicators.
# It's important to note that in this transformation, the original header row (representing the quarter names)
# in the wide format is not included as a data row in the long format.
# Such a transformation to a long format is less common in everyday data handling
# because it can make the dataset less immediately intuitive for human interpretation,
# as it consolidates multiple pieces of information into a denser format.
# ----- The Long Format of the Original Data -----
#                      Fiscal Quarters Indicators        Values
# 0                   Normalized Price  2Q FY2018  4.264000e+01
# 1       Price / Earnings - P/E (LTM)  2Q FY2018  1.640000e+01
# 2                    Net EPS - Basic  2Q FY2018  2.600000e+00
# 3                 Return On Equity %  2Q FY2018  4.086000e-01
# 4           Total Revenues / CAGR 5Y  2Q FY2018  7.910000e-02
# ..                               ...        ...           ...
# 225             Net Income / CAGR 5Y  4Q FY2023  1.026000e-01
# 226  Normalized Net Income / CAGR 5Y  4Q FY2023  9.300000e-02
# 227             Dividend Yield (LTM)  4Q FY2023  5.400000e-03
# 228            Market Capitalization  4Q FY2023  2.761224e+06
# 229                             Beta  4Q FY2023  1.290000e+00
#
# [230 rows x 3 columns]
