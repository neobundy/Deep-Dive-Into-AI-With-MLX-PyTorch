class Dish:
    def __init__(self, ingredient):
        self._ingredient = ingredient

    @property
    def ingredient(self):
        "The ingredient property."
        return self._ingredient

    @ingredient.setter
    def ingredient(self, value):
        if not value:
            raise ValueError("Ingredient cannot be empty")
        self._ingredient = value

# Using the class
my_dish = Dish("Tomato")
print(my_dish.ingredient)  # Accessing like an attribute

my_dish.ingredient = "Basil"  # Setting a new value
print(my_dish.ingredient)

# Trying to set an empty ingredient raises an error
try:
    my_dish.ingredient = ""
except ValueError as e:
    print(e)


# Tomato
# Basil
# Ingredient cannot be empty