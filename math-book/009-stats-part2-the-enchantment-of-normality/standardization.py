import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example data: Scores of students in two different subjects
data = {
    "Math_Scores": [88, 92, 76, 85, 82, 90, 72, 71, 93, 78],
    "Science_Scores": [72, 85, 78, 80, 75, 83, 69, 74, 88, 82]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Standardize the scores for comparison
df_standardized = (df - df.mean()) / df.std()

# Display the original and standardized scores
print("Original Scores:")
print(df)
print("\nStandardized Scores:")
print(df_standardized)

# Plotting the standardized scores to visualize
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df_standardized, fill=True)
plt.title('Standardized Scores Distribution')
plt.xlabel('Standardized Scores')
plt.ylabel('Density')
plt.legend(['Math Scores', 'Science Scores'])
plt.grid(True)
plt.show()

# Original Scores:
#    Math_Scores  Science_Scores
# 0           88              72
# 1           92              85
# 2           76              78
# 3           85              80
# 4           82              75
# 5           90              83
# 6           72              69
# 7           71              74
# 8           93              88
# 9           78              82
#
# Standardized Scores:
#    Math_Scores  Science_Scores
# 0     0.650145       -1.086012
# 1     1.140820        1.053103
# 2    -0.821881       -0.098728
# 3     0.282138        0.230366
# 4    -0.085868       -0.592370
# 5     0.895483        0.724008
# 6    -1.312557       -1.579654
# 7    -1.435226       -0.756918
# 8     1.263489        1.546745
# 9    -0.576544        0.559461