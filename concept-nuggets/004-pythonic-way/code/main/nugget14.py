# Creating a dictionary from a list of ingredients with their prices
ingredient_prices = [('flour', 1.5), ('sugar', 2), ('eggs', 0.3)]
prices_dict = {ingredient: price for ingredient, price in ingredient_prices}

print(prices_dict)
# {'flour': 1.5, 'sugar': 2, 'eggs': 0.3}