# Traditional approach: Adding spices to each ingredient one by one
spiced_ingredients = []
for ingredient in ['carrot', 'tomato', 'cucumber']:
    spiced_ingredients.append(ingredient + " with spice")

# Pythonic way: Preparing all ingredients with spices in one go using list comprehension
spiced_ingredients = [ingredient + " with spice" for ingredient in ['carrot', 'tomato', 'cucumber']]
print(spiced_ingredients)
# ['carrot with spice', 'tomato with spice', 'cucumber with spice']