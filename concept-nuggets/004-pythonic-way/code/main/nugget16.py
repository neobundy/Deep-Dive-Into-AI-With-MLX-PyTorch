from typing import List, Dict

def prepare_ingredients(ingredients: List[str]) -> Dict[str, int]:
    """
    Takes a list of ingredients and prepares them for cooking,
    returning a dictionary of ingredient names and their quantities.
    """
    prepared = {ingredient: len(ingredient) for ingredient in ingredients}
    return prepared

# Using the function with type hints
ingredients_list = ["tomatoes", "onions", "garlic"]
prepared_ingredients = prepare_ingredients(ingredients_list)
print(prepared_ingredients)
# {'tomatoes': 8, 'onions': 6, 'garlic': 6}