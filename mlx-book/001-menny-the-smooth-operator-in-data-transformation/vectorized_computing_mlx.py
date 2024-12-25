import mlx.core as mx

# Creating an MLX array
mlx_array = mx.array([1, 2, 3, 4, 5])

# Vectorized computation in MLX
mlx_result = mlx_array - 3

print(mlx_result)
# array([-2, -1, 0, 1, 2], dtype=int32)

