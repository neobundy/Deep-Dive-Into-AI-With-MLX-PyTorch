import mlx.core as mx

def square(x):
    return x * x

# Vectorizing the square function
vectorized_square = mx.vmap(square)

x = mx.array([1, 2, 3, 4])
squared_x = vectorized_square(x)
print("Before eval: ", squared_x)
mx.eval(squared_x)
print("After eval: ", squared_x)
# Before eval:  array([1, 4, 9, 16], dtype=int32)
# After eval:  array([1, 4, 9, 16], dtype=int32)
