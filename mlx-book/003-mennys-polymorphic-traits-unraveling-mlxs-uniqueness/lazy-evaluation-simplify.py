import mlx.core as mx

def duplicate_matrix_operations(x):
    y = x @ x  # Matrix multiplication
    z = x @ x  # Matrix multiplication again
    return y + z

x = mx.ones((10, 10))

# Without simplification
y = duplicate_matrix_operations(x)
print(y)    # Shows the result of y

# With simplification
z = duplicate_matrix_operations(x)
mx.simplify(z)  # Optimizes the graph to compute the matrix multiplication once
print(z)        # Shows the result of z after simplification
# array([[20, 20, 20, ..., 20, 20, 20],
#        [20, 20, 20, ..., 20, 20, 20],
#        [20, 20, 20, ..., 20, 20, 20],
#        ...,
#        [20, 20, 20, ..., 20, 20, 20],
#        [20, 20, 20, ..., 20, 20, 20],
#        [20, 20, 20, ..., 20, 20, 20]], dtype=float32)
# array([[20, 20, 20, ..., 20, 20, 20],
#        [20, 20, 20, ..., 20, 20, 20],
#        [20, 20, 20, ..., 20, 20, 20],
#        ...,
#        [20, 20, 20, ..., 20, 20, 20],
#        [20, 20, 20, ..., 20, 20, 20],
#        [20, 20, 20, ..., 20, 20, 20]], dtype=float32)
