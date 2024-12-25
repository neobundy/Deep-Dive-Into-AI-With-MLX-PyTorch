class RecipeBox:
    def __init__(self, recipes):
        self.recipes = recipes

    def __len__(self):
        return len(self.recipes)

# Creating a RecipeBox instance with a list of recipes
my_recipe_box = RecipeBox(["Pasta", "Pizza", "Salad"])

# Using the overridden __len__ method
print(f"The recipe box contains {len(my_recipe_box)} recipes.")
# The recipe box contains 3 recipes.