# Defining a generator to simulate fetching ingredients from a pantry
def fetch_ingredients():
    """Generator to simulate fetching ingredients from a pantry."""
    for ingredient in ['salmon', 'avocado', 'lemon']:
        yield ingredient

def cook(ingredient):
    """Simulates cooking the given ingredient."""
    print(f"Cooking {ingredient}...")

# Initialize the generator
ingredients_generator = fetch_ingredients()

# Efficiently processing ingredients with the walrus operator
while (ingredient := next(ingredients_generator, None)) is not None:
    cook(ingredient)
