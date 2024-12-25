import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_file = "./data/raw_data-aapl.csv"
image_name = "./data/normalized_price_histogram_seaborn.png"
target_column = 'Normalized Price'

# Load the dataset and transpose it
df = pd.read_csv(data_file, index_col=0).transpose()

# Now 'Normalized Price' can be accessed as a column
normalized_price = df[target_column]

# Set the style of seaborn
sns.set(style="whitegrid", color_codes=True, font_scale=1.2)

# Plot the histogram using seaborn
sns.histplot(normalized_price, bins=10, alpha=0.7, color='blue', kde=False)

# Set the labels and title
plt.title(f'Histogram of {target_column}')
plt.xlabel(target_column)
plt.ylabel('Frequency')

# Save the plot
plt.savefig(image_name)

# Show the plot
plt.show()

# Close the plot
plt.close()