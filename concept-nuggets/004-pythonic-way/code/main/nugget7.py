# Ingredients and their quantities
ingredients = ['flour', 'sugar', 'eggs', 'butter']
quantities = [200, 100, 2, 100]

# Combining the ingredients and quantities
for ingredient, quantity in zip(ingredients, quantities):
    print(f"{quantity}g of {ingredient}")

# 200g of flour
# 100g of sugar
# 2g of eggs
# 100g of butter