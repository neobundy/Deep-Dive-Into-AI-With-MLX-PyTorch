import mlx.core as mx

def square(x):
    return x * x

grad_square = mx.grad(square)

# Compute the gradient at x = 3
x = mx.array(3.0)
gradient_at_x = grad_square(x)
print("Before eval:", gradient_at_x)
mx.eval(gradient_at_x)
print("After eval:", gradient_at_x)
# Before eval: array(6, dtype=float32)
# After eval: array(6, dtype=float32)
