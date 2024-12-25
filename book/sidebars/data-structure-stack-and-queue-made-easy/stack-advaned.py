from collections import deque


class Stack:
    def __init__(self):
        self.container = deque()

    def push(self, item):
        self.container.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from an empty stack")
        return self.container.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from an empty stack")
        return self.container[-1]

    def is_empty(self):
        return not self.container

    def size(self):
        return len(self.container)


# Example usage:
stack = Stack()

# Push items onto the stack
stack.push('Tray 1')
stack.push('Tray 2')
stack.push('Tray 3')

print("Stack after push operations:", list(stack.container))

# Pop an item off the stack
popped_item = stack.pop()
print(f"Popped item: {popped_item}")
print("Stack after pop operation:", list(stack.container))

# Peek at the top item of the stack
top_item = stack.peek()
print(f"Top item: {top_item}")