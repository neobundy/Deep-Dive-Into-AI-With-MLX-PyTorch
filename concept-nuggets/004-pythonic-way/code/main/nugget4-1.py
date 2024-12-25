# Using a context manager to handle file operations
with open('recipe.txt', 'r') as file:
    contents = file.read()
    print(contents)

# The file is automatically opened and closed, no cleanup required