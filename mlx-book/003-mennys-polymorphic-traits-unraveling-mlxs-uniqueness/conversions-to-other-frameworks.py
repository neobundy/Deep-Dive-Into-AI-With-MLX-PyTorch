import mlx.core as mx
import numpy as np
import torch
import jax.numpy as jnp
import tensorflow as tf

# Create an MLX array
a_mlx = mx.arange(3)
print("Original MLX Array:", a_mlx)

# Convert MLX array to NumPy array and back
b_np = np.array(a_mlx)  # Convert to NumPy
c_mlx_from_np = mx.array(b_np)  # Convert back to MLX

# Create a NumPy array view
a_view_np = np.array(a_mlx, copy=False)
a_view_np[0] = 1

# Convert MLX array to PyTorch tensor and back
d_torch = torch.tensor(memoryview(a_mlx))
e_mlx_from_torch = mx.array(d_torch.numpy())

# Convert MLX array to JAX array and back
f_jax = jnp.array(a_mlx)
g_mlx_from_jax = mx.array(f_jax)

# Convert MLX array to TensorFlow tensor and back
h_tf = tf.constant(memoryview(a_mlx))
i_mlx_from_tf = mx.array(h_tf.numpy())

# Display the results
print("Converted to NumPy and Back:", c_mlx_from_np)
print("NumPy View Modified:", a_view_np[0].item(), "(Original MLX Array:", a_mlx[0].item(), ")")
print("Converted to PyTorch and Back:", e_mlx_from_torch)
print("Converted to JAX and Back:", g_mlx_from_jax)
print("Converted to TensorFlow and Back:", i_mlx_from_tf)

# Demonstrate the issue with gradients and external modifications
def modify_and_sum(x):
    x_view = np.array(x, copy=False)
    x_view[:] *= x_view  # External modification
    return x.sum()

x_mlx = mx.array([3.0])
y, df = mx.value_and_grad(modify_and_sum)(x_mlx)
print("Function Output (f(x) = x²):", y.item())
print("Gradient (Should be f'(x) = 2x, but is):", df.item())
