import numpy as np
# A list of mixed data types
a_list = [1, 2, 3, 4, 5, 'cat', 'dog', 'elephant']

# An array of integers
an_array = [1, 2, 3, 4, 5]

# An array of strings
another_array = ['cat', 'dog', 'elephant']

mixed_array = np.array(a_list)
integer_array = np.array(an_array)

# In Python, NumPy arrays can contain elements of different data types. When you create a NumPy array from a list that contains mixed data types, like a_list in your code, NumPy will choose a data type that can represent all the elements in the array. In this case, since a_list contains both integers and strings, NumPy will choose a data type that can represent both, which is a string.
print(mixed_array)
print(integer_array)

# ['1' '2' '3' '4' '5' 'cat' 'dog' 'elephant']
# [1 2 3 4 5]

print(type('1'))
print(type(1))

# <class 'str'>
# <class 'int'>