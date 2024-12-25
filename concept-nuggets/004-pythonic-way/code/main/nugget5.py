from collections import namedtuple

# Defining a named tuple for a culinary ingredient
Ingredient = namedtuple('Ingredient', ['name', 'quantity', 'unit'])

# Creating an instance of Ingredient
sugar = Ingredient(name='Sugar', quantity=100, unit='grams')

# Accessing the fields by name
print(f"Ingredient: {sugar.name}, Quantity: {sugar.quantity}{sugar.unit}")

# Ingredient: Sugar, Quantity: 100grams