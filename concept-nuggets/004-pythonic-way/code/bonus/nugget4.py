class Chef:
    def cook(self):
        return "The chef cooks a meal."

class Waiter:
    def serve(self):
        return "The waiter serves the meal."

class Restaurant:
    def __init__(self):
        self.chef = Chef()
        self.waiter = Waiter()

    def open_for_business(self):
        meal = self.chef.cook()
        serving = self.waiter.serve()
        return f"{meal} Then, {serving}"

# Using composition
restaurant = Restaurant()
print(restaurant.open_for_business())
# The chef cooks a meal. Then, the waiter serves the meal.