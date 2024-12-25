import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the CSV file
file_path = './data/raw_data-aapl.csv'

# Function to read data from file
def read_data(file_path):
    data = pd.DataFrame()
    data = pd.concat([data, pd.read_csv(file_path)], ignore_index=True)
    return data

# Read the data
stock_info = read_data(file_path)
print(stock_info.head())

# Melt the dataframe to long format suitable for seaborn plotting
stock_info_long = stock_info.melt(id_vars=['Fiscal Quarters'],
                                  var_name='Date',
                                  value_name='Value')

# Break down the long dataframe into individual dataframes for each metric
normalized_price_data = stock_info_long[stock_info_long['Fiscal Quarters'] == 'Normalized Price']
pe_ratio_data = stock_info_long[stock_info_long['Fiscal Quarters'] == 'Price / Earnings - P/E (LTM)']
roe_data = stock_info_long[stock_info_long['Fiscal Quarters'] == 'Return On Equity %']
cagr_data = stock_info_long[stock_info_long['Fiscal Quarters'] == 'Total Revenues / CAGR 5Y']

# Plot each metric using seaborn
for label, df in zip(['Normalized Price', 'P/E Ratio', 'ROE', 'CAGR'],
                     [normalized_price_data, pe_ratio_data, roe_data, cagr_data]):
    plt.figure(figsize=(10, 6))  # Set the figure size
    sns.lineplot(data=df, x='Date', y='Value').set_title(f'{label} over Time')
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.show()  # Show the plot

# Extract the columns for the correlation analysis
price = stock_info.loc[stock_info['Fiscal Quarters'] == 'Normalized Price'].drop('Fiscal Quarters', axis=1).iloc[0]
cagr = stock_info.loc[stock_info['Fiscal Quarters'] == 'Total Revenues / CAGR 5Y'].drop('Fiscal Quarters', axis=1).iloc[0]
roe = stock_info.loc[stock_info['Fiscal Quarters'] == 'Return On Equity %'].drop('Fiscal Quarters', axis=1).iloc[0]

# Create a new DataFrame with the extracted data
correlation_data = pd.DataFrame({'Stock Price': price, 'Revenue CAGR': cagr, 'ROE': roe})

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Print the correlation matrix
print(correlation_matrix)

# Alternatively, to visualize the correlation matrix, you can use seaborn's heatmap function
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Analysis between Stock Price, Revenue CAGR, and ROE")
plt.show()

from scipy.stats import pearsonr

# Pearson correlation between metrics
# Let's take the relevant rows for each metric from the previous 'stock_info' DataFrame.
# Make sure you have your DataFrame 'stock_info' prepared as before.

# Get the values for each metric
price_values = stock_info.loc[stock_info['Fiscal Quarters'] == 'Normalized Price'].iloc[0, 1:].astype(float)
cagr_values = stock_info.loc[stock_info['Fiscal Quarters'] == 'Total Revenues / CAGR 5Y'].iloc[0, 1:].astype(float)
roe_values = stock_info.loc[stock_info['Fiscal Quarters'] == 'Return On Equity %'].iloc[0, 1:].astype(float)

# Perform Pearson correlation test between Stock Price and Revenue CAGR
price_cagr_corr, price_cagr_p_value = pearsonr(price_values, cagr_values)

# Perform Pearson correlation test between Stock Price and ROE
price_roe_corr, price_roe_p_value = pearsonr(price_values, roe_values)

# Perform Pearson correlation test between Revenue CAGR and ROE
cagr_roe_corr, cagr_roe_p_value = pearsonr(cagr_values, roe_values)

# Print out the correlation coefficients and p-values
print(f"Stock Price and Revenue CAGR Pearson correlation: {price_cagr_corr} (p-value: {price_cagr_p_value})")
print(f"Stock Price and ROE Pearson correlation: {price_roe_corr} (p-value: {price_roe_p_value})")
print(f"Revenue CAGR and ROE Pearson correlation: {cagr_roe_corr} (p-value: {cagr_roe_p_value})")

# Note: P-values are used to determine the significance of the results.
# A p-value less than 0.05 is often considered to indicate strong evidence against the null hypothesis (i.e., a correlation exists).