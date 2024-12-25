# A diverse list of ingredients, including both sweet and savory types
ingredients = ["sugar", "salt", "pepper", "vanilla", "cinnamon", "tomato", "basil"]

# Traditional approach: Conditionally replacing only the sweet ingredients, leaving savory ones untouched
adjusted_ingredients = []
for ingredient in ingredients:
    if ingredient in ["sugar", "vanilla", "cinnamon"]:
        adjusted_ingredients.append("garlic")  # A savory replacement
    else:
        adjusted_ingredients.append(ingredient)

# Pythonic way: Using list comprehension with conditions for a sophisticated ingredient adjustment
adjusted_ingredients = [
    "garlic" if ingredient in ["sugar", "vanilla", "cinnamon"] else ingredient for ingredient in ingredients
    if ingredient not in ["basil"]  # Excluding basil from being replaced
]
print(adjusted_ingredients)
# ['garlic', 'salt', 'pepper', 'garlic', 'garlic', 'tomato']
