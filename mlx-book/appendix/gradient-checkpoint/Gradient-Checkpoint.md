# Gradient Checkpointing in MLX - Introduced in v0.1.0

![title..png](images%2Ftitle..png)

Gradient checkpointing in MLX introduces a smart approach to managing memory consumption during the training or fine-tuning of large models. This technique, which involves selectively saving intermediate activations during the forward pass and recomputing them as needed during the backward pass, provides a significant advantage in terms of memory efficiency. By doing so, it offers a balance between computational overhead and memory usage, a trade-off that is especially beneficial when working with large models on hardware where memory is at a premium.

Imagine you're working on a huge puzzle, but your table is too small to lay out all the pieces at once. To solve the puzzle, you decide to work on small sections at a time. You pick a section, assemble it, take a picture with your phone (to remember how it looks), and then put the pieces back in the box. When you need to connect this section to another, you refer to your picture, pull out the pieces again, and fit them together.

This is essentially what gradient checkpointing in MLX does with large neural network models during training. Training a model involves two main passes: a forward pass, where the model makes predictions, and a backward pass, where it updates itself based on errors in those predictions. During these passes, the model performs many calculations (the "puzzle pieces"), which require memory to store.

Normally, to update the model (solve the puzzle), you'd keep all these calculations (puzzle pieces) out on the table (in memory) during both the forward and backward passes. However, this can take up a lot of space, especially with large models.

Gradient checkpointing offers a clever workaround. During the forward pass (the first attempt at solving the puzzle), instead of keeping all intermediate calculations (all puzzle pieces) in memory, it saves just a few key pieces (takes pictures of key sections). Then, during the backward pass (when connecting everything together), it uses these saved pieces to recompute the necessary calculations (re-assembles sections of the puzzle as needed) instead of pulling them from memory. This means you don't need as large a table (as much memory) because you're not keeping every single piece out at once.

This technique trades a bit of extra work (recomputing some calculations) for a significant reduction in memory usage, making it possible to train larger models (solve bigger puzzles) on hardware with limited memory. It's a strategic choice that allows for more efficient use of resources, particularly beneficial when dealing with complex models that would otherwise require more memory than is available.

### Simplified Explanation of Gradient Checkpointing with `@mx.checkpoint` Decorator

![hike.png](images%2Fhike.png)

Think of gradient checkpointing in MLX like a smart camera feature during a scenic hike. Instead of taking photos of every single view (which fills up your phone's storage quickly), you mark certain spots (using `@mx.checkpoint`) where you'll take a photo. Later, if you want to recall the entire hike, you can use these key photos to help reconstruct the memories without needing to store every single view.

In MLX, the `@mx.checkpoint` decorator works similarly for functions during model training. It tells MLX, "Hey, don't remember everything this function does during the first walk-through (forward pass). Just save the starting points (inputs). If we need details about what happened here later (during the backward pass for gradient calculation), we'll just redo this part of the hike (recompute the function's outputs from the saved inputs) instead of trying to remember everything."

This method significantly reduces how much information (memory) we need to keep during training, especially for large models.

```python
import mlx.core as mx

@mx.checkpoint  # This is like marking a scenic spot for a photo.
def computeExpSum(inputs):
    # This function calculates the sum of the exponentials of the inputs.
    return mx.sum(mx.exp(inputs))

# Define some input data.
inputs = mx.array([1.0, 2.0, 3.0], dtype='float32')

# Calculate the gradient of our function with respect to its inputs, using gradient checkpointing.
gradient = mx.grad(computeExpSum)(inputs)  # Uses less memory during computation.

# Here, 'gradient' tells us how changes in 'inputs' affect changes in the output of 'computeExpSum',
# but we saved memory by not keeping track of every detail in 'computeExpSum' during the forward pass.
```

In this code snippet, `computeExpSum` is a function that sums the exponentials of its inputs. By decorating it with `@mx.checkpoint`, we're instructing MLX to only save the inputs (`inputs`) during the forward pass and not the entire sequence of calculations. This way, if we need to understand how to adjust `inputs` to minimize errors (in the backward pass), we only recompute the necessary parts using the saved inputs, significantly reducing memory usage.

The decorator concept works well in the scenario of gradient checkpointing in MLX for several reasons:

1. **Seamless Integration**: Decorators in Python allow for the modification of functions or methods without altering their code directly. By using the `@mx.checkpoint` decorator, gradient checkpointing can be applied transparently to any function, enabling or disabling this feature without changing the function's internal implementation. This makes it easy to experiment with memory optimization techniques on different parts of a model's computations.

2. **Flexibility**: Decorators provide a flexible way to add or remove functionality. In the context of MLX, the `@mx.checkpoint` decorator can be applied to selective functions where memory efficiency is desired, without affecting the rest of the code. This selective application is crucial for optimizing memory usage without incurring unnecessary computational overhead everywhere.

3. **Simplicity and Readability**: Using a decorator simplifies the implementation of complex functionality like gradient checkpointing. Instead of wrapping function calls or manually managing the storage and recomputation of intermediate activations, developers can just annotate functions with `@mx.checkpoint`. This keeps the code clean and readable, making it easier to understand and maintain.

4. **Automatic Management of Computational Graphs**: Gradient checkpointing involves sophisticated manipulation of the computational graph, including saving inputs and not intermediate activations, and later recomputing these activations during the backward pass. The decorator abstracts away these complexities, automatically managing the modifications to the computational graph required for checkpointing. This allows developers to leverage gradient checkpointing without needing to understand the intricate details of computational graph manipulation.

5. **Optimization Without Sacrificing Functionality**: The primary goal of gradient checkpointing is to reduce memory usage during the training of large models. By employing a decorator, MLX allows for this optimization while ensuring that the original functionality of the function (i.e., the computations it performs) remains unchanged. This means that models can be trained on hardware with limited memory without compromising the model's ability to learn from data.

In essence, the decorator concept is ideal for implementing gradient checkpointing because it offers a non-intrusive, flexible, and user-friendly means of enhancing memory efficiency in deep learning models. This approach allows developers and researchers to focus on designing and improving their models rather than getting bogged down by the underlying complexities of memory management.

### Applying Checkpointing in Models

Gradient checkpointing can also be seamlessly integrated into model architectures, such as in the `Transformer` model, by utilizing the `checkpoint` parameter at strategic points like the transformer encoder layers. When set to `checkpoint=True`, these layers employ gradient checkpointing during the forward pass, saving only essential information for recomputing activations during the backward pass. This significantly lowers the memory footprint when training complex models, like those found in transformer architectures, by reducing the memory required to store intermediate activations.

In `mlx.nn.Transformer`:

```python

class Transformer(Module):
    """
    Implements a standard Transformer model.

...
        chekpoint (bool, optional): if ``True`` perform gradient checkpointing
            to reduce the memory usage at the expense of more computation.
            Default: ``False``.
    """

    def __init__(
        self,
        dims: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Callable[[Any], Any] = relu,
        custom_encoder: Optional[Any] = None,
        custom_decoder: Optional[Any] = None,
        norm_first: bool = True,
        checkpoint: bool = False,
    ):
        super().__init__()

        self.encoder = custom_encoder or TransformerEncoder(
...
            checkpoint,
        )

        self.decoder = custom_decoder or TransformerDecoder(
...
            checkpoint,
        )

    def __call__(self, src, tgt, src_mask, tgt_mask, memory_mask):
        memory = self.encoder(src, src_mask)
        return self.decoder(tgt, memory, tgt_mask, memory_mask)

```

![drawing.png](images%2Fdrawing.png)

Let's simplify the explanation with a more straightforward analogy:

Imagine you're creating a big, detailed drawing, but you have a small desk and not enough space to keep all your drawing tools, sketches, and colors out at once. To manage, you decide to use only a few tools at a time and store the rest. Every time you need a tool you previously put away, you bring it back out, use it, and then store it again. This way, you manage to create your big drawing without needing a bigger desk.

In the world of training complex models like Transformers, gradient checkpointing works somewhat similarly. These models perform a lot of calculations (like using many drawing tools), which usually requires a lot of memory (a big desk) to store the results of these calculations (the sketches and colors). But memory can be expensive or limited.

By using gradient checkpointing, specifically the `checkpoint=True` setting in parts of the model like the Transformer's encoder layers, we're telling the model, "Hey, don't keep all the results of your calculations at once. Just save the really important stuff (like key sketches). If we need the rest, we'll figure it out again later (bring back out the tools and redo some sketches)." This means the model uses less memory because it's not trying to hold onto every single calculation it makes. Instead, it only keeps the essentials and re-does some work if needed.

`checkpoint=True` is like deciding which tools and sketches to keep out on your small desk. This smart memory management allows training bigger and more complex models (creating a bigger drawing) without needing more memory (a bigger desk).

### Benefits and Trade-offs

The principal advantage of gradient checkpointing lies in its ability to drastically reduce memory usage, thus enabling the training of more substantial models or employing larger batch sizes within the constraints of existing hardware. However, this efficiency comes at the expense of increased computational demand, as certain activations must be recomputed during the backward pass. This can extend training times but is often a worthwhile trade-off for the memory savings achieved.

### Integrating Gradient Checkpointing into the Archery Model

![bullseye.png](images%2Fbullseye.png)

As we delve back into the Archery example from Chapter 3 of my first book, "Foundations of Neural Networks - A Comprehensive Overview," we explore a significant enhancement: the integration of gradient checkpointing.

[Chapter 3 - Foundations of Neural Networks - A Comprehensive Overview](..%2F..%2F..%2Fbook%2F003-foundations-of-neural-networks%2FREADME.md)

The Archery Model, inspired by the remarkable precision of Korean archers who routinely strike the bullseye, mirrors the exceptional skills of these Olympic medalists. As a Korean, the prowess and consistency of these archers not only fill me with immense pride but also serve as a metaphor for the precision and efficiency we strive for in neural network training.

Here's the diff between the original `archery-model.py` and the updated version `archery-model-with-gradient-checkpoint.py`:

![diff-archery-model.png](images%2Fdiff-archery-model.png)

Contrasting the original `archery-model.py` with the updated version, we pinpoint the pivotal change: the implementation of gradient checkpointing. This modification is not realized through the use of a decorator but is instead integrated directly into the model's architecture, demonstrating an effective strategy for memory management during training.

The adjustment to incorporate gradient checkpointing is a strategic decision aimed at optimizing memory usage. By selecting specific points within the model to apply this technique, such as after the activation function, we manage to substantially reduce the memory footprint. This is achieved without the explicit use of the `mx.checkpoint` decorator, contrary to the example provided earlier. Instead, the checkpointing is directly integrated into the model's workflow, showcasing the flexibility and adaptability of gradient checkpointing in MLX.

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class ArcheryModel(nn.Module):
    def __init__(self):
        super(ArcheryModel, self).__init__()
        self.layer = nn.Linear(1, 1)  # Single input to single output
        self.activation = nn.ReLU()  # ReLU Activation Function

    def __call__(self, x):
        x = self.layer(x)
        x = mx.checkpoint(self.activation)(x)  # Apply checkpointing directly in the call
        return x

model = ArcheryModel()

def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

optimizer = optim.SGD(learning_rate=0.01)

train_data = [(mx.array([0.5]), mx.array([0.8]))]

for epoch in range(100):
    for input, target in train_data:
        loss, grads = loss_and_grad_fn(model, input, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

Gradient checkpointing is directly integrated into the Archery Model's workflow. Specifically, it is applied within the `__call__` method, where the activation function's output is passed through `mx.checkpoint`. This technique reduces memory consumption by not storing intermediate activations for the entire duration of the forward and backward passes. Instead, it stores minimal necessary information and recomputes the activations during the backward pass as required.

This approach underscores the practical application of gradient checkpointing in managing memory efficiently, particularly in training complex models on hardware with constrained memory resources. Through this example, we demonstrate how subtle modifications can yield significant improvements in memory management, enhancing the feasibility of training more sophisticated models within limited hardware environments.

## Notes on the MLX Book Appendix

Lexy (my one and only go-to MLX whiz, powered by GPT-4) and I took a deep dive into gradient checkpointing in MLX, aiming to make it as clear as possible for you. With not much in the way of official documentation, we relied on our knowledge and insights into MLX's potential.

A quick heads-up: we might not have nailed everything perfectly. If we spot any mistakes or if new information comes to light, we're on it – we'll update this guide to keep it fresh.

Think of the MLX Appendix as a living document, evolving alongside MLX itself. We'll be keeping our eyes peeled, ready to refresh and revise as MLX continues to grow and change.

