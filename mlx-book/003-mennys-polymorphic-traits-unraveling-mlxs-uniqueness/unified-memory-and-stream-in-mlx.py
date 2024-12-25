import mlx.core as mx

def fun(a, b, d1, d2):
    # Perform a matrix multiplication on stream d1
    x = mx.matmul(a, b, stream=d1)  # Heavy lifting

    # Perform a series of exponentiations on stream d2
    for _ in range(100):
        b = mx.exp(b, stream=d2)  # Many small tasks

    return x, b

# Initialize data
a = mx.random.uniform(shape=(4096, 512))
b = mx.random.uniform(shape=(512, 4))

# Specify the streams (devices) for the operations
device1 = mx.gpu  # Assuming the heavy task is better on GPU
device2 = mx.cpu  # Assuming the smaller tasks are better on CPU

# Run the function with specified streams
result_x, result_b = fun(a, b, device1, device2)

# Optional: Evaluate the results (if needed)
mx.eval(result_x, result_b)

print(result_x)
print(result_b)
