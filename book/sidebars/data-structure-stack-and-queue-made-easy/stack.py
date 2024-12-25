# A simple stack implementation using a Python list

stack = []

# Push items onto the stack
stack.append('Tray 1')
stack.append('Tray 2')
stack.append('Tray 3')

print("Stack after push operations:", stack)

# Pop an item off the stack
# Added a condition to check for an empty stack before popping
if stack:
    popped_item = stack.pop()
    print(f"Popped item: {popped_item}")
else:
    print("Cannot pop from an empty stack.")
print("Stack after pop operation:", stack)

# Peek at the top item of the stack
# Added a condition to check for an empty stack before peeking
if stack:
    top_item = stack[-1]
    print(f"Top item: {top_item}")
else:
    print("Cannot peek on an empty stack.")