import pandas as pd

# Each row represents a metric and each column represents a fiscal quarter. In our case, rows are features and columns are observations.
data = {
    'Fiscal Quarters': ['Normalized Price', 'Price / Earnings - P/E (LTM)', 'Net EPS - Basic', 'Return On Equity %'],
    '2Q FY2018': [42.64, 16.4, 2.6, 0.4086],
    '3Q FY2018': [47.816, 17.2, 2.78, 0.4537],
    '4Q FY2018': [55.8, 18.6, 3, 0.4936],
    '1Q FY2019': [39.168, 12.8, 3.06, 0.4605]
    # ... other fiscal quarters
}

# Creating the DataFrame in the original wide format
df_wide = pd.DataFrame(data)
print("DataFrame in Wide Format:")
print(df_wide)

# The DataFrame needs to be transposed so that fiscal quarters are rows and metrics are columns
df_long = df_wide.set_index('Fiscal Quarters').transpose()

# Now each row is a fiscal quarter and each column is a metric
print("DataFrame in Long Format:")
print(df_long)
