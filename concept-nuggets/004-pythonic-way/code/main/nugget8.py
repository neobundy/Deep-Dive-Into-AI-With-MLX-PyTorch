from functools import reduce

# Ingredients represented by their nutritional value
nutritional_values = [100, 200, 300, 400]

# Map: Enhance each value (like adding spices)
enhanced_values = list(map(lambda x: x * 1.1, nutritional_values))

# Filter: Select values above 250 (like choosing the ripest fruits)
filtered_values = list(filter(lambda x: x > 250, enhanced_values))

# Reduce: Combine values to get a total (like mixing ingredients into a dish)
total_nutritional_value = reduce(lambda x, y: x + y, filtered_values)

print(f"Enhanced Values: {enhanced_values}")
print(f"Filtered Values: {filtered_values}")
print(f"Total Nutritional Value: {total_nutritional_value}")

# Enhanced Values: [110.00000000000001, 220.00000000000003, 330.0, 440.00000000000006]
# Filtered Values: [330.0, 440.00000000000006]
# Total Nutritional Value: 770.0