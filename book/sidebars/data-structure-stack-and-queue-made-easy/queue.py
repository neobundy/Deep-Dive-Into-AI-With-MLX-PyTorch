# A simple queue implementation using a Python list

queue = []

# Enqueue items to the back of the queue
queue.append('Customer 1')
queue.append('Customer 2')
queue.append('Customer 3')

print("Queue after enqueue operations:", queue)

# Dequeue an item from the front of the queue
# Added a condition to check for an empty queue before dequeuing
if queue:
    dequeued_item = queue.pop(0)
    print(f"Dequeued item: {dequeued_item}")
else:
    print("Cannot dequeue from an empty queue.")
print("Queue after dequeue operation:", queue)

# Peek at the first item in the queue
# Added a condition to check for an empty queue before peeking
if queue:
    first_item = queue[0]
    print(f"First item in queue: {first_item}")
else:
    print("Cannot peek on an empty queue.")