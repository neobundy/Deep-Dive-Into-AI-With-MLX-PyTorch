ingredient = 'Garlic'
quantity = 3

# Traditional string formatting
description = '{} cloves of {}'.format(quantity, ingredient)

# Pythonic way with f-strings
description = f"{quantity} cloves of {ingredient}"
print(description)
# 3 cloves of Garlic