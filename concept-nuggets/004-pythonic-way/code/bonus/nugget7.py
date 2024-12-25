from dataclasses import dataclass

@dataclass
class Recipe:
    name: str
    ingredients: list
    prep_time: int

# Instantiating a Recipe object
chocolate_cake = Recipe("Chocolate Cake", ["flour", "sugar", "cocoa powder", "eggs", "milk", "butter"], 45)

print(chocolate_cake)
# Recipe(name='Chocolate Cake', ingredients=['flour', 'sugar', 'cocoa powder', 'eggs', 'milk', 'butter'], prep_time=45)