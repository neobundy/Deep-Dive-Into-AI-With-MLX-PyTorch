# Defining a decorator to sprinkle 'magic' on any dish (function)
def sprinkle_magic(func):
    def wrapper():
        print("Sprinkling magic dust...")
        func()
        print("Dish is now enchanted!")
    return wrapper

# Applying the decorator to a function
@sprinkle_magic
def prepare_dish():
    print("Preparing the dish.")

# Serving the enchanted dish
prepare_dish()

# Defining a decorator for preprocessing and cleanup
def kitchen_prep_and_cleanup(func):
    def wrapper(*args, **kwargs):
        print("Prepping ingredients...")
        result = func(*args, **kwargs)  # Execute the function
        print("Cleaning up the kitchen...")
        return result
    return wrapper

# Applying the decorator to a cooking function
@kitchen_prep_and_cleanup
def cook_dish(dish_name):
    print(f"Cooking {dish_name} to perfection.")

# Cooking a dish with preprocessing and cleanup
cook_dish("vegetable stir-fry")