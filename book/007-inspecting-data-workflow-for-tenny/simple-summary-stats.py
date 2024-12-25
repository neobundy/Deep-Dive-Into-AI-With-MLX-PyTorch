import pandas as pd

# Sample data: exam scores of students
data = {'Scores': [88, 92, 80, 89, 90, 78, 85, 91, 76, 94]}

# Creating a DataFrame
df = pd.DataFrame(data)

# Calculating basic summary statistics
mean = df['Scores'].mean()
median = df['Scores'].median()
mode = df['Scores'].mode()[0]  # mode() returns a Series
std_dev = df['Scores'].std()
min_score = df['Scores'].min()
max_score = df['Scores'].max()
count = df['Scores'].count()
quantiles = df['Scores'].quantile([0.25, 0.5, 0.75])

# Displaying the summary statistics
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev}")
print(f"Min Score: {min_score}")
print(f"Max Score: {max_score}")
print(f"Count: {count}")
print(f"Quantiles:\n{quantiles}")