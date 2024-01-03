# Composable Function Transformations

JAX is a Python library that supercharges mathematical and scientific computations. It's like a turbocharged version of NumPy, a popular tool for numerical operations. JAX stands out because it can execute calculations very quickly, especially on powerful hardware like GPUs and TPUs. It's particularly adept at tasks like finding derivatives automatically, a crucial aspect in fields like machine learning. What makes JAX unique is its ability to transform mathematical functions in various ways, enhancing their performance or providing new ways to understand them. 

MLX is known to closely follow existing frameworks like NumPy, PyTorch and JAX. 

It is stated in the official documentation:

    MLX is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon, brought to you by Apple machine learning research.
    
    The Python API closely follows NumPy with a few exceptions. MLX also has a fully featured C++ API which closely follows the Python API.
    
    The main differences between MLX and NumPy are:
    
    - Composable function transformations: MLX has composable function transformations for automatic differentiation, automatic vectorization, and computation graph optimization.
    
    - Lazy computation: Computations in MLX are lazy. Arrays are only materialized when needed.
    
    - Multi-device: Operations can run on any of the supported devices (CPU, GPU, …)
    
    The design of MLX is inspired by frameworks like PyTorch, Jax, and ArrayFire. A noteable difference from these frameworks and MLX is the unified memory model. Arrays in MLX live in shared memory. Operations on MLX arrays can be performed on any of the supported device types without performing data copies. Currently supported device types are the CPU and GPU.


As shown in the  [hello-ai-world-mlx.py](../../000-hello-ai-world/hello-ai-world-mlx.py) example
MLX is different in how gradients are computed. Unlike PyTorch, where you typically compute the gradients by calling the `backward()` method on a loss tensor, MLX uses a different approach that is similar to JAX. In MLX, you use the `value_and_grad` function to create a new function that, when called, computes both the value of the original function and its gradient with respect to its parameters. This is what is meant by "composable function transformations".

PyTorch:

```python

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 5000
for epoch in range(num_epochs):
    predictions = model(x_train_tensor)
    loss = loss_function(predictions, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```

MLX:

```python

    def loss_fn(model, x, y):
        return nn.losses.mse_loss(model(x), y)

```

In this line, `nn.value_and_grad(model, loss_fn)` returns a new function `loss_and_grad_fn`. When you call this function with some inputs, it computes both the value of `loss_fn` and its gradients with respect to the parameters of model. This approach is more functional and composable, as it allows you to create complex functions by combining simpler ones, and automatically computes gradients for you.

The term "composable" in this context refers to the ability to combine simple functions to create more complex ones. This is a key feature of functional programming.  In the context of MLX and similar libraries like JAX, "composable function transformations" means that you can take a function that computes some value (like a loss function), and easily create a new function that computes both the value and its gradients. This is done using the `value_and_grad` function.

This approach allows you to build complex computations by combining simple, reusable pieces, and the library automatically handles the computation of gradients for you. This makes it easier to build and reason about complex models and algorithms, especially in the context of machine learning and optimization where gradient computations are a key part.

Composable function transformations are a powerful feature of JAX, allowing you to combine different transformations like jit (Just-In-Time compilation), grad (gradient), vmap (vectorization), and others in a flexible way. This **composability** enables efficient and concise expression of complex operations. Here's an example to illustrate this:

Suppose we have a simple function f that performs some calculations. We want to compute its gradient, compile it for better performance, and vectorize it to apply it over an array of inputs. Here's how you can do this in JAX:

Define a basic function:

```python
def f(x):
    return x ** 2 + x * 3
```

Compute the gradient:

```python
from jax import grad
df = grad(f)
```

JIT compile the gradient function:

```python
from jax import jit
df_jit = jit(df)
```

Vectorize the JIT compiled gradient function:

```python
from jax import vmap
df_jit_vectorized = vmap(df_jit)
```

Apply the composed function:

```python
import jax.numpy as jnp
x_values = jnp.array([1.0, 2.0, 3.0, 4.0])
results = df_jit_vectorized(x_values)
print(results)
```

In this example, `df_jit_vectorized` is a composition of multiple transformations: it computes the gradient of f, compiles this computation for better performance, and then vectorizes it to apply over an array of inputs. When you run this, you'll get the gradients of f evaluated at each point in `x_values`, with the whole computation being efficiently executed.

This demonstrates the power and elegance of composable transformations, enabling complex operations to be written in a concise and readable manner.

MLX follows this philosophy of composable function transformations, allowing you to build complex models and algorithms by combining simple, reusable pieces. This makes it easier to reason about and understand the code, and also enables efficient execution on hardware like GPUs.


```python
def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=learning_rate)

num_epochs = 5000

for epoch in range(num_epochs):
    loss, grads = loss_and_grad_fn(model, x_train_tensor, y_train_tensor)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.5f}')
```

For the benefits of vectorized computation, refer to the [Vectorized_Computation.md](../vectorized-computation/Vectorized_Computation.md) sidebar.

As evident, the principles of inheritance and polymorphism from object-oriented programming can be metaphorically applied to enhance your learning experience. Embracing this essence of object orientation in your daily life encourages you to build upon previous knowledge and adapt it flexibly to new contexts.