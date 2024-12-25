import mlx.core as mx

def softmax(x):
    """Compute softmax in MLX."""
    e_x = mx.exp(x - mx.max(x))
    return e_x / e_x.sum(axis=0)

# Example usage
scores = mx.array([3.0, 1.0, 0.2])
print(softmax(scores))
print(sum(softmax(scores)))