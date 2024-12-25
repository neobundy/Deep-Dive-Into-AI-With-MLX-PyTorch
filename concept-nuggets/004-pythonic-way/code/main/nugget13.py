from collections import Counter

ingredient_list = ['egg', 'flour', 'egg', 'butter', 'flour', 'flour']

# Counting the ingredients
ingredient_count = Counter(ingredient_list)
print(ingredient_count)
# Counter({'flour': 3, 'egg': 2, 'butter': 1})