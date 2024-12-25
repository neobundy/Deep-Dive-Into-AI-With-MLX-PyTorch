import mlx.core as mx

# Indexing and Slicing
# Creating an array (like laying out a deck of cards)
arr = mx.arange(10)

# Picking specific items (cards) from the array
third_item = arr[3]      # Picks the 4th card
second_last_item = arr[-2] # Picks the second last card
selected_range = arr[2:8:2] # Picks every 2nd card from 3rd to 8th

# Multi-Dimensional Indexing
# Creating a multi-dimensional array (stack of card decks)
multi_arr = mx.arange(8).reshape(2, 2, 2)

# Using ':' and '...' for multi-dimensional indexing
first_column = multi_arr[:, :, 0] # Picks the first card from every mini-deck
first_column_ellipsis = multi_arr[..., 0] # Same as above

# Adding a New Dimension
new_dim_arr = arr[None] # Adds a new deck on top

# Advanced Indexing with Another Array
idx = mx.array([5, 7])  # Index array
indexed_arr = arr[idx]  # Picks the 6th and 8th cards based on idx

# In-Place Updates
a = mx.array([1, 2, 3])
a[2] = 0  # Changing the 3rd card to 0

# Linking Arrays
b = a
b[2] = 0  # Change reflected in both a and b

# Saving a Single Array
single_deck = mx.array([1.0])
mx.save("single_deck", single_deck)

# Loading a Single Array
loaded_single_deck = mx.load("single_deck.npy")

# Saving Multiple Arrays
b = mx.array([2.0])
mx.savez("multi_decks", a, b=b)

# Loading Multiple Arrays
loaded_decks = mx.load("multi_decks.npz")

# Displaying Results
print("Third Item:", third_item)
print("Second Last Item:", second_last_item)
print("Selected Range:", selected_range)
print("First Column:", first_column)
print("First Column with Ellipsis:", first_column_ellipsis)
print("Array with New Dimension:", new_dim_arr.shape)
print("Indexed Array:", indexed_arr)
print("In-place Updated Array:", a)
print("Linked Array b:", b)
print("Loaded Single Deck:", loaded_single_deck)
print("Loaded Deck 'b':", loaded_decks['b'])
