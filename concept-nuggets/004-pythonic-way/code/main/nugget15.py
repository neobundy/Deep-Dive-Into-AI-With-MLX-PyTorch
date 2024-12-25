from pathlib import Path

# Creating a Path object for the recipes directory
recipes_path = Path('recipes')

# Checking if the directory exists and creating it if not
if not recipes_path.exists():
    recipes_path.mkdir()
    print("The recipes directory has been created!")

# Creating a new recipe file within the directory
(recipes_path / 'chocolate_cake.txt').write_text("Chocolate Cake Recipe Ingredients...")

print("Recipe saved successfully!")