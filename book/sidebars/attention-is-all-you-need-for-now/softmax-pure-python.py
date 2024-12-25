import math

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = [math.exp(i) for i in x]
    return [j / sum(e_x) for j in e_x]

# Example usage
scores = [3.0, 1.0, 0.2]
print(softmax(scores))
print(sum(softmax(scores)))