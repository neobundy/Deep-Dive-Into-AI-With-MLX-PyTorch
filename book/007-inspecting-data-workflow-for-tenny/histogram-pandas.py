import pandas as pd
import matplotlib.pyplot as plt

data_file = "./data/raw_data-aapl.csv"
image_name = "./data/normalized_price_histogram_pandas.png"
target_column = 'Normalized Price'

# Load the dataset and transpose it
df = pd.read_csv(data_file, index_col=0).transpose()

# Now 'Normalized Price' can be accessed as a column
normalized_price = df[target_column]

# Plot the histogram
plt.hist(normalized_price, bins=10, alpha=0.7, color='blue')
plt.title(f'Histogram of {target_column}')
plt.xlabel(target_column)
plt.ylabel('Frequency')
plt.grid(True)
# Save the plot
plt.savefig(image_name)
plt.show()
plt.close()
