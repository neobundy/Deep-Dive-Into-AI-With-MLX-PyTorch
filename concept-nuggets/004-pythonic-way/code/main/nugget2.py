# Traditional approach: Creating a list of squared numbers, consuming more space
squared_numbers = [number ** 2 for number in range(10)]

# Pythonic way: Using a generator to create squared numbers, saving memory
squared_numbers_generator = (number ** 2 for number in range(10))

# Serving the squared numbers one at a time
for squared_number in squared_numbers_generator:
    print(squared_number)


# Generator function to yield a sequence of spiced dishes
def spiced_dish_sequence(spices):
    for spice in spices:
        # Preparing each dish with a unique spice
        yield f"Dish seasoned with {spice}"

# Defining a list of spices for our dishes
spice_list = ["saffron", "cinnamon", "cardamom", "turmeric"]

# Using the generator function to prepare each dish on-demand
for dish in spiced_dish_sequence(spice_list):
    print(dish)