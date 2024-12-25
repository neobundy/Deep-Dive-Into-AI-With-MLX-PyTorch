# A simple list of dishes' prices
dish_prices = [10, 20, 15, 30]

# Using a lambda function to apply a discount (the chef's secret touch)
discounted_prices = list(map(lambda price: price * 0.9, dish_prices))

print(discounted_prices)
# [9.0, 18.0, 13.5, 27.0]