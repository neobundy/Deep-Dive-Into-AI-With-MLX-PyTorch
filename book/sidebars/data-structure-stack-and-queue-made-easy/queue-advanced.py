from collections import deque


class Queue:
    def __init__(self):
        self.container = deque()

    def enqueue(self, item):
        self.container.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        return self.container.popleft()

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from an empty queue")
        return self.container[0]

    def is_empty(self):
        return not self.container

    def size(self):
        return len(self.container)


# Example usage:
queue = Queue()

# Enqueue items into the queue
queue.enqueue('Customer 1')
queue.enqueue('Customer 2')
queue.enqueue('Customer 3')

print("Queue after enqueue operations:", list(queue.container))

# Dequeue an item from the queue
dequeued_item = queue.dequeue()
print(f"Dequeued item: {dequeued_item}")
print("Queue after dequeue operation:", list(queue.container))

# Peek at the first item in the queue
first_item = queue.peek()
print(f"First item in queue: {first_item}")