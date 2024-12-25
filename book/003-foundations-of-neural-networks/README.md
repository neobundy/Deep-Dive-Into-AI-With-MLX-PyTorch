# Chapter 3 Foundations of Neural Networks - A Comprehensive Overview

Neural networks, inspired by the structure and functions of the human brain, have revolutionized the field of machine learning. These powerful computational models are designed to mimic the way neurons in our brain interact, making them incredibly effective at processing complex patterns in data. From recognizing speech to driving autonomous vehicles, neural networks have wide-ranging applications.

The journey of a neural network's learning process begins with the _forward pass_, where input data is fed through the network to produce an output. However, the learning doesn't stop there. _Backpropagation_, a cornerstone of neural network training, involves tweaking the network’s _parameters_ (_weights_ and _biases_) based on the output's error. This process is complemented by _gradient descent_, an optimization technique that minimizes the error by adjusting the weights in the direction that most reduces the loss. These elements work in concert to refine the network’s performance, making it more accurate over time.

_Activation functions_ are the heartbeat of neural networks. They decide whether a neuron should be activated or not, essentially determining the output based on the input received. These functions add non-linear properties to the network, enabling it to learn complex patterns. From sigmoid to ReLU, different activation functions have unique characteristics that influence the network’s behavior and performance.

_Loss functions_ play the role of a guide in the learning journey of a neural network. They measure the difference between the network’s prediction and the actual target. By quantifying the extent of error, loss functions such as Mean Squared Error or Cross-Entropy provide a clear objective for the network to improve upon during training.

_Optimizers_ are the navigators in the world of neural networks. They influence how quickly and effectively a network learns by adjusting the weights and _learning rate_. Optimizers like SGD, Adam, and RMSprop each have unique ways of ensuring that the network converges to the minimum loss, balancing the speed of learning with the risk of overshooting the lowest point of error.

As neural networks delve deeper, they face challenges like _exploding_ and _dying gradients_, where gradients become too large or too small to propagate useful information through the network. This section explores strategies like gradient clipping and batch normalization to counter these issues, ensuring a stable and efficient learning process.

Understanding these fundamental aspects of neural networks opens the door to leveraging their full potential in various applications, making them a cornerstone of modern artificial intelligence.

As we embark on this journey through the intricate world of neural networks, we'll delve into each of these fundamental concepts in detail. But before we dive into the mechanics of neural networks, let's confront a profound truth about our universe: the absence of perfect circles.

## The Quest for Perfection in an Imperfect Universe

![terminator-modelling.png](terminator-modelling.png)

There is no perfect circle in our universe.

At first glance, this statement might seem bewildering. But here's the crux of the matter: in our universe, the notion of a perfect circle is more an ideal than a tangible reality. The circles we encounter in nature or create ourselves are approximations of this concept. These shapes, which we often perceive as flawlessly round, are in fact subtle deceptions — either tricks of nature or constructs of our human perception. 

This realization sets the stage for understanding neural networks. Just as the pursuit of perfect circles leads us to a deeper appreciation of geometry and physics, exploring the complexities of neural networks reveals insights into artificial intelligence and machine learning. Let's journey through this world, embracing the imperfections and learning to find the beauty and potential within them.

For those not acquainted with 3D modeling, it's interesting to note that any 3D object is constructed using vertices and edges. A vertex is a point in space, and an edge is a line connecting two vertices. To form the simplest plane figure, a triangle, you need at least three vertices. These three vertices together are referred to as a face, which is a plane figure bounded by straight edges. Faces can take various shapes like triangles, squares, pentagons, hexagons, and so on.

I suggest experimenting with Blender, a free open-source 3D software, to see this in action.

![blender1.png](blender1.png)

In Blender, for instance, you'll notice that what appears to be a circle is actually composed of multiple vertices, say 32. To create something that more closely resembles a round circle, you might need upwards of 64 vertices. The shape starts to look circular when it has about 8 vertices, but even then, it's not truly a perfect circle - it's an approximation, an illusion created by the proximity of the vertices.

This illustrates a fundamental truth: in our universe, there are no perfect circles, only approximations made up of vertices and edges.

When discussing straight lines, it's clear that a line drawn between any two points, or vertices, is straight. This is an observable fact, evident even without mathematical proof.

The concept of slope in mathematics is intrinsically linked to the idea of a line. A line is essentially defined by its slope. In the formula `y = mx + b`, `m` represents the slope of the line, while `b` denotes the y-intercept, the point where the line crosses the y-axis. This simple equation encapsulates the relationship between a line and its slope, a fundamental concept in geometry and algebra.

In the realm of neural networks, `y` represents the output, `x` is the input, `m` corresponds to the weight, and `b` is the bias. The process of training a neural network involves determining the optimal `m` and `b` values for given `x` and `y` inputs. During each training epoch or iteration, the network employs a trial-and-error approach to find the best `m` and `b`. The network's accuracy is assessed using a loss function, a mathematical formula that calculates the error, which is the disparity between the predicted output (`y hat`) and the actual value (`y`).

The notation `y hat` is used to denote the predicted value, the output generated by the neural network, while `y` stands for the actual value. This distinction is necessary to differentiate between the predicted and actual values in equations and analyses:

```
y_hat = mx + b
```

For example, if a given data point `x = 10` should ideally yield `y = 20`, but the neural network predicts `y hat = 15`, then the error is 5. This error is computed by the loss function, which measures the difference between `y hat` and `y`:

```
loss = y_hat - y
```

In Python programming, it's crucial to distinguish between variable names and mathematical expressions. For instance, `mx` is interpreted as a variable name, whereas `m * x` represents the multiplication of `m` and `x`. Thus, the asterisk `*` is used for multiplication:

```python
y = 10
y_hat = m * x + b

# If y_hat is 15, then the loss is 5
loss = y_hat - y
```

This approach is straightforward, emphasizing the importance of clarity and accuracy in representing mathematical formulas in programming languages like Python.

In the pursuit of minimizing loss, neural networks continually iterate their process, aiming for a loss value of zero or as close as possible. This optimization is guided by various factors, including predetermined epochs or a set threshold for the loss.

But how exactly does a neural network determine the best method for learning to minimize this loss? This is where the gradient descent algorithm plays a crucial role, requiring an understanding of differential calculus.

The essence of finding a slope in mathematics lies in calculating the ratio of the change in y (vertical change) to the change in x (horizontal change). This is represented mathematically as `m = Δy / Δx`, where `Δ` (delta) signifies change. In Python, this can be expressed as `m = (y2 - y1) / (x2 - x1)`.

In linear equations, such as `y = 2x + 1`, the slope (here, 2) is the coefficient of x, and the y-intercept is the constant term (here, 1). In the context of neural networks, these correspond to the weight and bias, respectively. The slope represents how y changes with x; for example, an increase in x by 1 leads to an increase in y by 2.

Lines are straightforward, but what about curves? At first glance, finding the slope of a curve seems impossible, as it's not a straight line. However, a curve is essentially composed of numerous straight line segments. By zooming in on a curve and selecting two very close points, one can approximate the slope at that section. Despite the curve appearing smooth and round, it's essentially made up of straight lines connecting vertices.

Thus, the flawless forms that appear to populate our cosmos are fashioned in such a manner. They are, in a manner of speaking, normalized to exude perfection to our observing eyes.

![blender4.png](blender4.png)
![blender5.png](blender5.png)
![blender6.png](blender6.png)
![blender7.png](blender7.png)
![blender8.png](blender8.png)
![blender9.png](blender9.png)

Normalization, a concept relevant both mathematically and philosophically, involves adjusting values measured on different scales to a common scale. For more insights on normalization, refer to "Normalization-Made-Easy.md".

[Normalization-Made-Easy.md](..%2Fsidebars%2Fnormalization-made-easy%2FNormalization-Made-Easy.md)

![graphs.png](graphs.png)

In mathematical terminology, a linear equation forms a straight line when graphed (`y = mx + b`), while non-linear equations form curves. These include polynomial equations of various degrees, such as quadratic (`y = x^2 + 2x + 1`), cubic (`y = x^3 + x^2 + 2x + 1`), and quartic equations (`y = x^4 + x^3 + 2x + 1`).

Use this example code to plot the graphs of these equations in Python:

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data range
x = np.linspace(-10, 10, 400)

# Linear equation: y = mx + b
y_linear = 2 * x + 3

# Quadratic equation: y = x^2 + 2x + 1
y_quadratic = x**2 + 2*x + 1

# Cubic equation: y = x^3 + x^2 + 2x + 1
y_cubic = x**3 + x**2 + 2*x + 1

# Quartic equation: y = x^4 + x^3 + 2x + 1
y_quartic = x**4 + x**3 + 2*x + 1

# Plotting
plt.figure(figsize=(12, 10))

# Linear
plt.subplot(2, 2, 1)
sns.lineplot(x=x, y=y_linear)
plt.title('Linear Equation')

# Quadratic
plt.subplot(2, 2, 2)
sns.lineplot(x=x, y=y_quadratic)
plt.title('Quadratic Equation')

# Cubic
plt.subplot(2, 2, 3)
sns.lineplot(x=x, y=y_cubic)
plt.title('Cubic Equation')

# Quartic
plt.subplot(2, 2, 4)
sns.lineplot(x=x, y=y_quartic)
plt.title('Quartic Equation')
plt.show()
```

You need matplotlib and seaborn libraries to run this code. If you don't have them installed, you can use the following commands to install them:

```bash
pip install matplotlib, seaborn
```

Or just run this command to install all the required libraries for this book:
    
```bash
pip install -r requirements.txt
``` 

## Trial and Triumph: Forward Pass, Backpropagation, Gradient Descent

There's no need to be an expert in differential calculus to comprehend the fundamental concept of loss minimization in neural networks. At its core, the network's mission is to pinpoint the deepest valley along a curve, much like how a person without sight might use a cane to navigate to the lowest part of a slope. This exploration encompasses both _forward passes_, a method of exploration and error, and _backpropagation_, the network's strategy of adapting from previous errors. Often, backpropagation is informally abbreviated to _backprop_. The method of seeking the deepest trough through such means is known as _gradient descent_, which entails determining the gradient at each juncture and moving towards the sharpest downward gradient.

The network's aspiration is to perpetually hone its adjustments, journeying toward this point of minimal error through an iterative process that mirrors the dynamics of trial and error. This learning odyssey unfolds through both forward and backward passes. In the vein of an ideal human learner—though it's well to note that not every individual exemplifies this paradigm—it absorbs lessons from its errors, diligently pursuing advancement with each successive iteration, or _epoch_.

For a deeper conceptual understanding, consider reading "A-Path-to-Perfection-AI-vs-Human.md". This essay delves into the nuances of AI learning and human parallels, offering a unique perspective on the journey towards achieving perfection in neural networks.

[A-Path-to-Perfection-AI-vs-Human.md](..%2F..%2Fessays%2FAI%2FA-Path-to-Perfection-AI-vs-Human.md)

When you delve into the workings of a neural network, it's easy to get caught up in the simplicity of scalar values. But the reality is far more complex, as the network is juggling matrices and vectors. It's like a dance of numbers and calculations in multi-dimensional space. And in this dance, the gradient descent algorithm plays a crucial role, guiding the network to find the most effective slope.

Now, when we speak of the parameters of a neural network, we're essentially talking about its building blocks - the weights and biases. These are the critical values that the network tweaks and tunes as it learns. The entire process is a quest to discover the perfect settings for these parameters, a journey made possible through the gradient descent algorithm.

```python
parameters = weights + biases  # This line sums up the essence of neural network parameters.
```

Think of a neural network as a complex structure made up of layers, each filled with neurons. Each neuron is equipped with its own set of weights and biases. For the numerically gifted, calculating the number of parameters in a neural network might seem like an intriguing challenge. However, for most, utilizing the tools provided by deep learning frameworks such as PyTorch or MLX is a far more practical approach.

In a neural network, each neuron in a layer is connected to neurons in the previous layer through weighted connections. These weights are parameters that the network learns during training. The weight signifies the strength of the connection between neurons. Along with weights, neural networks often use biases. A bias is an additional parameter associated with each neuron and is added to the weighted sum of inputs before applying the _activation function_, which will be discussed later. The purpose of the bias is to provide each neuron the flexibility to shift the activation function to the left or right, which can be crucial for successful learning.

So, in a typical fully connected (dense) layer of a neural network, each neuron has its own set of weights and a single bias. For instance, in a layer with three neurons, each receiving inputs from two neurons in the previous layer, there would be a total of 6 weights (2 per neuron) and 3 biases (1 per neuron). That is, each neuron has its own set of weights and a single bias. 

To illustrate this concept, let's look at examples of a simple neural network in both PyTorch and MLX. This network will consist of a single layer with three neurons. In such a setup, the number of parameters (weights and biases) depends on the input size. For this example, let's assume the input size is 2.

In PyTorch, you define a model by creating a class that inherits from `nn.Module`. Let's define a single-layer network with 3 neurons:

```python
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, num_neurons):
        super(SimpleNetwork, self).__init__()
        self.layer = nn.Linear(input_size, num_neurons)

    def forward(self, x):
        return self.layer(x)

# Creating the network
input_size = 2
num_neurons = 3
model = SimpleNetwork(input_size, num_neurons)

# Counting parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
# Total number of parameters: 9
```

`model.parameters()` method returns an iterator over all parameters (weights and biases) of the model, both learnable and fixed. In `p.numel()`, `numel()` is a method of `torch.Tensor` objects. It stands for "number of elements" and returns the total number of elements in the tensor. For example, if a parameter tensor `p` has a shape of `[3, 2]` (like a weight matrix connecting a layer with 2 neurons to a layer with 3 neurons), `p.numel()` would return 6, since there are 6 elements (weights) in total in the matrix.

Finally, `sum(...)` sums up the numbers returned by `p.numel()` for each parameter tensor in the model. This yields the total number of parameters in the neural network.

So, `total_params = sum(p.numel() for p in model.parameters())` calculates the total count of individual weight and bias elements in the entire neural network. This is often used to get a sense of the size and complexity of the model, as more parameters typically mean a larger, more complex model.

This network will have a total of 9 parameters - 6 weights (2 inputs * 3 neurons) and 3 biases (1 for each neuron).

### MLX Example

In MLX:

```python
import mlx.core as mx
import mlx.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, num_neurons):
        super().__init__()
        self.layer = nn.Linear(input_size, num_neurons)

    def __call__(self, x):
        return self.layer(x)

# Create and initialize the network
input_size = 2
num_neurons = 3
model = SimpleNetwork(input_size, num_neurons)
mx.eval(model.parameters())

# Accessing parameters
params = model.parameters()

# Counting parameters
total_params = 0
for layer in params.values():
    for param in layer.values():
        total_params += param.size

print(f"Total number of parameters: {total_params}")
```

The model is created with a single linear layer that has 3 neurons. The `model.parameters()` method provides the parameters in a nested dictionary format. To count the total number of parameters, you iterate through each layer and then through each parameter (weights and biases) in the layer. The `param.size` attribute gives you the total number of elements in each parameter array, which you sum up to get the total number of parameters for the model.

In both PyTorch and MLX, the `nn.Linear` (or equivalent) layer creates a set of weights and biases. The total number of parameters is the sum of all these weights and biases. For a network with a single layer and three neurons receiving two inputs, there will be 6 weights and 3 biases, resulting in 9 parameters.

## Sparking Connections: Understanding Activation Functions

The role of activation functions in neural networks and the importance of non-linearity is a fundamental concept in understanding how neural networks function and learn complex patterns. 

Activation functions in neural networks decide whether a neuron should be activated or not. Just like a neuron in human biology, which fires only when a certain threshold is reached, a neuron in a neural network is activated if the sum of weighted inputs exceeds a certain value. This decision is based on the weighted sum of the inputs plus a bias. By applying a non-linear transformation to the input signal, activation functions allow neural networks to compute and learn complex functions, which can be applied to complex tasks like language translation, image recognition, and more.

If neural networks only used linear transformations (like `y = mx + b`), they would not be able to solve problems beyond simple linear regression. The reason is that the composition of two linear functions is still a linear function. No matter how many layers you stack in a neural network, if all are linear, the entire network will still perform a linear transformation, lacking the ability to capture complex patterns in data. 

Non-linear functions (`y = f(x)`, where `f` is non-linear), on the other hand, enable the network to learn non-linear mappings from inputs to outputs. This is crucial for tasks like classifying images, where the relationship between features is not linear.

This concept is so crucial that it's worth exploring in greater detail through the lens of a familiar and straightforward game: "Tic-Tac-Toe". A notable depiction of this game can be found in the movie 'War Games' (1983). In the film, the protagonist, played by Matthew Broderick, inadvertently engages a military supercomputer named WOPR (War Operation Plan Response) in a game of global thermonuclear war. WOPR, also programmed to play Tic-Tac-Toe, simulates all possible outcomes of the game. It quickly realizes that Tic-Tac-Toe invariably ends in a draw with optimal play, illustrating the concept of a "no-win scenario."

In Tic-Tac-Toe, players alternate marking 'X' or 'O' in a 3x3 grid, aiming to align three of their marks in a row, either horizontally, vertically, or diagonally. Despite its simplicity, the game captures a fundamental principle: the inherent limitations of deterministic, linear strategies.

A neural network limited to linear transformations (akin to `y = mx + b`) in playing Tic-Tac-Toe might only grasp the most direct and basic tactics, such as choosing the center square or blocking an immediate win. However, Tic-Tac-Toe's ultimate lesson, particularly at the highest level of play, is that linear strategies lead to predictable and unvarying outcomes — typically a draw.

The more profound insight here, especially relevant to neural networks, is the necessity of non-linear processing for complex decision-making. If the neural network were to employ non-linear transformations (similar to non-linear activation functions), it could theoretically learn more nuanced strategies. These might include setting traps or predicting the opponent's future moves, representing a deeper understanding of the game's dynamics.

However, as 'War Games' poignantly illustrates, even with advanced, non-linear strategies, Tic-Tac-Toe remains a deterministic game. The complexity and variability are inherently limited. This underscores a pivotal point in neural network design: the need for non-linear capabilities to address complex, non-deterministic problems. In real-world applications, unlike Tic-Tac-Toe, challenges are rarely black and white and often require the ability to discern intricate patterns and navigate uncertain outcomes.

While non-linear processing in neural networks is essential for complex problem-solving, the Tic-Tac-Toe analogy in 'War Games' elegantly highlights the limitations of deterministic systems and the importance of adapting strategies to the nature of the problem at hand.

Let's now turn to activation functions in neural networks, which are the key to enabling non-linear processing.

### Types of Activation Functions

Common non-linear activation functions include:

1. **ReLU (Rectified Linear Unit)**:
   - **Definition**: `f(x) = max(0, x)`.
   - **Characteristics**: Linear (identity) for all positive values and zero for all negative values.
   - **Use Cases**: Widely used in hidden layers of neural networks, particularly effective in classification and regression models, and deep learning architectures.

2. **LeakyReLU**:
   - **Definition**: `f(x) = x if x > 0 else alpha * x` (where `alpha` is a small, non-zero constant).
   - **Characteristics**: Similar to ReLU, but allows a small, non-zero gradient when the unit is inactive (i.e., for negative values).
   - **Use Cases**: Often used in scenarios where ReLU may not perform well due to the "dying ReLU" problem. LeakyReLU can help mitigate this by allowing a small gradient when the unit is not active, making it useful in deep learning networks where dead neurons are a concern.
     - The "dying ReLU" problem refers to an issue encountered in neural networks that use the Rectified Linear Unit (ReLU) activation function.This problem occurs when a ReLU neuron gets stuck by only outputting zero for all inputs. In other words, if a neuron's output becomes zero, it starts only passing zero values through the network during forward propagation. Consequently, during backpropagation, the gradient through that neuron is also zero, which means the weights connected to that neuron are not updated during training. The dying ReLU problem often arises when neurons are exposed to large, negative inputs. This can shift the neuron into a state where it only outputs zero, essentially "killing" the neuron for all inputs during the training process. Once a ReLU neuron dies, it stops contributing to the learning process because it doesn't activate on any data point. This can lead to reduced capacity of the model to learn complex patterns in the data, as effectively, the neuron is removed from the network. A common solution is to use variants of the ReLU function, such as Leaky ReLU or Parametric ReLU (PReLU), which allow a small, positive gradient when the unit is inactive. This modification helps to keep the neurons alive throughout the training process.

3. **Sigmoid**:
   - **Definition**: `f(x) = 1 / (1 + exp(-x))`.
   - **Characteristics**: Maps input values between 0 and 1.
   - **Use Cases**: Commonly used in binary classification models, especially as the output layer for predicting probabilities.

4. **Tanh (Hyperbolic Tangent)**:
   - **Definition**: `f(x) = (2 / (1 + exp(-2x))) - 1`.
   - **Characteristics**: Maps input values between -1 and 1.
   - **Use Cases**: Often used in hidden layers of neural networks for classification and regression, effective for modeling inputs with strong negative, neutral, and strong positive values.

5. **Softmax**:
   - **Definition**: Converts a vector of values into a probability distribution.
   - **Characteristics**: Outputs sum to 1; each output is proportional to the probability of belonging to a particular class.
   - **Use Cases**: Primarily used in the output layer of neural networks for multi-class classification tasks, such as digit recognition or image classification.

Each of these activation functions has unique properties making them suitable for different types of neural network models and tasks, particularly in the realm of classification. LeakyReLU, in particular, addresses some limitations of ReLU, offering an alternative in situations where neurons might otherwise become inactive and stop contributing to the learning process.

When you encounter new terms like these, take a moment to explore their etymology. You might be surprised by what you discover. 

The names of these activation functions in neural networks are derived from their mathematical properties or the shapes of their graphs:

1. **ReLU (Rectified Linear Unit)**:
   - "Rectified" means "adjusted," and "Linear Unit" refers to the function's linear behavior for positive input values.
   - ReLU is called so because it rectifies (or clips) negative values to zero and keeps positive values unchanged, thus acting as a linear function for positives and a zero function for negatives.

2. **LeakyReLU (Leaky Rectified Linear Unit)**:
   - Similar to ReLU, it rectifies negative values, but instead of setting them to zero, it allows a small, non-zero, constant gradient (hence "leaky").
   - The term "Leaky" implies that this function leaks a bit of the values (typically very small) even when the input is negative, unlike the ReLU, which blocks (rectifies to zero) all negative values.

3. **Sigmoid**:
   - The name "sigmoid" comes from the Greek letter sigma (σ), due to its S-shaped curve.
   - The sigmoid function is a type of logistic function that maps any value into a range between 0 and 1, forming an S-shaped curve.

4. **Tanh (Hyperbolic Tangent)**:
   - "Tanh" is short for hyperbolic tangent, a mathematical function.
   - It’s called so because it represents a hyperbolic version of the trigonometric tangent function. The graph of tanh is similar to sigmoid but stretched to fit within the range of -1 to 1.

5. **Softmax**:
   - The term "Softmax" combines the words "soft" and "maximum." It's named this way because it softly picks the largest value in a set, turning it into probabilities. The "soft" part means it doesn't just pick the biggest number outright but gives other numbers a chance based on their size. The sum of all the probabilities is 1. 
   - It’s called "soft" because it provides a "softer" classification than a hard, direct maximum. The Softmax function outputs a probability distribution - it’s like choosing the maximum probability (hence "maximum") but in a way that allows for some degree of uncertainty or "softness."

Each name of these activation functions effectively captures their mathematical and visual characteristics. The power of words is evident here, as the names chosen for these neural network functions not only describe their behavior but also reflect the thought processes of the researchers who created them. These names provide insight into the development and progression of neural network technology.

The choice of activation function depends on the specific application and properties of the network being designed. Without these non-linear activation functions, neural networks would not have achieved the widespread success and versatility they enjoy in a vast array of complex tasks across different domains.

It's essential to choose the right activation functions when designing neural networks. Without any activation function, a neural network essentially becomes a linear regression model, lacking the ability to capture complex patterns. Similarly, using an inappropriate activation function can significantly hinder performance or prevent the model from learning effectively. For instance, using a softmax activation function in a logistic regression model, which typically requires a sigmoid function, would lead to unexpected and incorrect results.

Even if you use an incorrect activation function in a model, it might not result in outright errors, but the model's performance will suffer. The outcomes will be inaccurate, and the model's learning ability will be compromised. Essentially, this would render the model ineffective or, in simpler terms, make it "dumb."

## Learning from Loss: The Role of Loss Functions

Loss functions in neural networks play a pivotal role in the learning process. They are essentially mathematical formulas used to quantify how far the network's predictions are from the desired outcomes. The choice of a loss function depends on the specific task at hand – whether it's a regression problem, classification problem, or something else. Understanding this concept can be made easier with a simple analogy.

### Conceptual Example: Archery

![bullseye.png](bullseye.png)

Have you ever marveled at the remarkable precision of Korean archers in consistently striking the bullseye? Achieving perfect scores is almost routine for these Olympic medalists. Yes, as a Korean, this fills me with pride!

Their success goes beyond just possessing a steady hand and sharp eyesight. It's equally about the ability to learn and adapt from each and every misstep.

Think of training a neural network like teaching someone to play archery. 

- **Target**: The bullseye on the archery target represents the actual output or the _ground truth_ that your neural network aims to achieve.
- **Arrows**: Each arrow shot at the target is akin to a prediction made by the neural network.
- **Distance from Bullseye**: The distance of each arrow from the bullseye can be thought of as the error in the network’s prediction. The further the arrow lands from the bullseye, the greater the error.
- **Loss Function**: In this analogy, the loss function is like the rules that measure how far off each shot is from the bullseye. For instance, it might be a simple rule like measuring the straight-line distance from the bullseye to where the arrow lands.

### Training Process

- **Learning**: Just as an archer adjusts their aim based on where previous arrows landed, the neural network adjusts its parameters (weights and biases). This is done in an attempt to make the next shot – or in the case of the network, the next prediction – closer to the bullseye.
- **Minimizing the Loss**: The goal is to get the arrows as close to the bullseye as possible. In neural network terms, this means adjusting the weights and biases to minimize the loss function. The smaller the loss, the closer the network's predictions are to the actual values.
- **Iterative Process**: Just like an archer needs several attempts to perfect their aim, a neural network undergoes many iterations of forward passes (making predictions) and backpropagation (updating parameters based on the loss) to minimize the loss function.

The loss function in a neural network determines how well the network is performing. It quantifies the error in predictions, and the training process involves minimizing this error. Just as a Korean archer learns to hit the bullseye more consistently with practice, a neural network learns to make more accurate predictions by iteratively adjusting its parameters to reduce the loss.

Let's create a simple example in both PyTorch and MLX that mimics the archery analogy for training a neural network, including the loss and activation functions. We'll build a very basic model for a regression task where the target (like hitting the bullseye) is a specific numeric value.

In PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple Neural Network
class ArcheryModel(nn.Module):
    def __init__(self):
        super(ArcheryModel, self).__init__()
        self.layer = nn.Linear(1, 1)  # Single input to single output
        self.activation = nn.ReLU()   # ReLU Activation Function

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

# Instantiate the model
model = ArcheryModel()

# Loss Function
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example training data: [input, target]
train_data = [(torch.tensor([0.5]), torch.tensor([0.8]))]

# Training Loop
for epoch in range(100):  # Number of times to iterate over the dataset
    for input, target in train_data:
        # Forward pass
        output = model(input)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

Recall that MLX uses _composable functions_. In MLX, the concept of _composable function transformations_ refers to a programming approach where functions are designed to take another function as input and then return a new function as output. This method allows for more flexible and efficient code, as it enables the combination or "composition" of various functions into a single operation.

For example, in MLX, you might have a function that calculates the loss of a neural network and another function that computes the gradient. With composable function transformations, you can combine these two into a single operation that both calculates the loss and its gradient. This not only streamlines the process but also aligns well with the principles of functional programming, leading to clearer and more concise code. 

For more on this, refer to this sidebar:
[Composable-Function-Transformations.md](..%2Fsidebars%2Fcomposable-function-transformations%2FComposable-Function-Transformations.md)

In MLX:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Simple Neural Network
class ArcheryModel(nn.Module):
    def __init__(self):
        super(ArcheryModel, self).__init__()
        self.layer = nn.Linear(1, 1)  # Single input to single output
        self.activation = nn.ReLU()   # ReLU Activation Function

    def __call__(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

# Instantiate the model
model = ArcheryModel()

# Define a loss function
def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)

# Composable function for loss and gradient
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Optimizer
optimizer = optim.SGD(learning_rate=0.01)

# Example training data: [input, target]
train_data = [(mx.array([0.5]), mx.array([0.8]))]

# Training Loop
for epoch in range(100):  # Number of times to iterate over the dataset
    for input, target in train_data:
        # Forward pass and loss calculation
        loss, grads = loss_and_grad_fn(model, input, target)

        # Backward pass and optimization
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

Unlike PyTorch, in MLX, the use of composable function transformations is a key feature:

- We define a custom loss function `loss_fn` which applies the mean squared error loss.
- `loss_and_grad_fn` is a composable function that calculates both the loss and the gradients in one step.
- During each iteration of the training loop, `loss_and_grad_fn` is called to perform a forward pass and compute the gradients simultaneously.
- The optimizer then updates the model parameters based on these gradients.

This approach aligns with MLX's design for efficient and clear computation, where functions can be composed together for streamlined execution.

Now, you will see how Korean archers can inspire the design of neural networks.

### Common Loss Functions

Here are some common loss functions, their use cases, and etymology:

#### 1. Mean Squared Error (MSE)
- **Use Case**: Commonly used in regression problems, where the task is to predict continuous values. For instance, predicting house prices based on features like size and location.
- **Etymology**: "Mean" refers to the average, "Squared" indicates each error term (difference between predicted and actual value) is squared, and "Error" refers to the inaccuracies in predictions.

#### 2. Cross-Entropy Loss
- **Use Case**: Widely used in classification problems, especially binary classification like email spam detection (spam or not spam) and multi-class classification like image classification (identifying objects in images).
- **Etymology**: "Cross" signifies the interplay between the predicted probability distribution and the actual distribution, and "Entropy" is a concept from information theory that measures the amount of information or uncertainty.

#### 3. Mean Absolute Error (MAE)
- **Use Case**: Also used in regression tasks. It’s less sensitive to outliers compared to MSE. An example would be predicting the age of a person based on features like height and weight.
- **Etymology**: "Mean" denotes the average, "Absolute" implies the absolute value of the errors (ignoring the direction of errors), and "Error" represents the prediction inaccuracies.

#### 4. Hinge Loss
- **Use Case**: Often employed in binary classification problems, especially with Support Vector Machines (SVMs). For example, in facial recognition (identifying if a face is person A or not).
- **Etymology**: The term "Hinge" is used because the function is shaped like a door hinge, designed to 'swing' one way for correct classifications and another for incorrect ones.

#### 5. Log Loss
- **Use Case**: Typically used in binary classification problems, like predicting whether a tumor is benign or malignant. It emphasizes correct prediction of the actual class.
- **Etymology**: Derived from "Logarithmic" loss, as it involves the logarithm of the predicted probabilities. It quantifies the accuracy of a classifier by penalizing false classifications.

Each of these loss functions has its unique properties and is chosen based on the specific requirements and nature of the machine learning problem being solved. The names usually give an insight into the mathematical operation or the core concept behind the function.

## Navigating Pitfalls: Addressing Exploding & Dying Gradients

In the context of neural networks and machine learning, gradients are essentially vectors of partial derivatives. They represent the rate of change of a function (like a loss function) with respect to its parameters (like weights in a neural network). Gradients play a crucial role in the training of neural networks through a process called gradient descent as detailed earlier, where they guide how the network's parameters should be adjusted to minimize the loss function.

You will hear the term '_grads_' more often than its full form, '_gradients_.' This is often used as a shorthand for gradients. It's a more concise way to refer to these derivative vectors, especially in code and mathematical formulas. In many programming frameworks like PyTorch, TensorFlow, or MLX, you'll often see variables named 'grads' in the context of backpropagation and optimization algorithms. This shorthand helps maintain readability and simplicity in code, especially when dealing with complex mathematical operations.

The exploding gradients problem occurs during the training of a neural network when the gradients (the values used to update the network's weights) become very large. This often happens in deeper networks with many layers.

### Exploding Gradients Problem

When gradients explode, they can cause the learning process to become unstable. The model's weights might update too drastically during training, leading to an erratic learning process and poor model performance.

To prevent exploding gradients, normalization techniques like gradient clipping are used. Gradient clipping involves setting a threshold value, and if the gradient exceeds this value, it's "clipped" or reduced to keep it within a manageable range. This ensures stable and effective training.

See this sidebar for more on normalization in general:

[Normalization-Made-Easy.md](..%2Fsidebars%2Fnormalization-made-easy%2FNormalization-Made-Easy.md)

Pause for a moment to consider the prevalence of logarithmic functions in the context of neural networks, particularly when it comes to the task of normalization. Essentially, when values become excessively large or diminutive, logarithms serve the critical purpose of re-scaling them to a more manageable spectrum. Indeed, this concept is a fundamental one in the realm of data science, where logarithmic functions are applied with precision to normalize values. They ensure that figures stay anchored within a range that is both practical for analysis and functional for computational processes.

### Dying Gradients Problem

The dying gradients problem is the opposite of exploding gradients. It occurs when the gradients become too small, effectively "dying." This often happens when using certain activation functions like ReLU (Rectified Linear Unit).

When gradients die, the weights of certain parts of the neural network stop updating. This means these parts of the network can't learn from the data, leading to poor performance of the model.

One common way to address dying gradients is to use different activation functions, such as LeakyReLU, which prevents the gradients from dying out. Batch normalization can also be helpful. It normalizes the input layer by adjusting and scaling the activations, which helps maintain healthy gradients throughout the network.

### Normalization Techniques

Normalization techniques are crucial in dealing with both exploding and dying gradients. They modify the inputs or gradients to be within a certain range, which makes the training process more stable and efficient. 

- **Gradient Clipping**: This is used to prevent exploding gradients by limiting the size of the gradients to a defined range.
- **Batch Normalization**: This technique normalizes the inputs of each layer, ensuring that the network behaves in a more predictable and stable manner, which is beneficial for addressing the dying gradients issue.

In summary, the exploding and dying gradients problems are significant challenges in training deep neural networks. They can lead to instability and inefficiency in the learning process. Utilizing normalization techniques like gradient clipping and batch normalization can effectively mitigate these issues, leading to more stable and successful training of neural network models.

If you're aiming to normalize your grasp of normalization, consider diving into the world of logarithmic concepts. They'll significantly bolster your understanding.

## Parameters vs. Hyperparameters

Parameters are mostly learnable while hyperparameters are mostly tunable. 

_Parameters_ are the internal variables of the model that are learned from the training data. For example, in a linear regression model, the weights and biases are parameters. 

_Hyperparameters_, on the other hand, are the settings of the training process that are set before training starts and are not learned from the data. Examples of hyperparameters include the learning rate, the number of epochs (iterations over the entire dataset), and the batch size.

So, parameters are "**learnable**" because they are learned from the data during training, while hyperparameters are "**tunable**" because they are manually set and adjusted by the developer to optimize the learning process and the performance of the model.

### Parameters: Learnable Aspects

1. **Linear Regression Model**: 
   - **Parameters**: These are the weights (coefficients) and the bias (intercept) of the model. 
   - **How They're Learned**: During training, the model adjusts these weights and bias to fit the training data as closely as possible. For instance, in a simple linear equation `y = mx + b`, `m` and `b` are the parameters. The model learns the optimal values of `m` (slope) and `b` (y-intercept) based on the data.

2. **Neural Network**:
   - **Parameters**: In a neural network, the parameters are the weights and biases of each neuron in the layers of the network.
   - **How They're Learned**: These are learned through backpropagation and gradient descent methods. As the network is exposed to training data, it continuously adjusts these weights and biases to minimize the error in its predictions.

### Hyperparameters: Tunable Settings

1. **Learning Rate**:
   - **Hyperparameter**: This determines the size of the steps the model takes during gradient descent.
   - **Tuning**: A developer might start with a learning rate of 0.01, for example, and adjust it based on whether the model is learning too slowly (indicating a need for a higher rate) or overshooting (requiring a lower rate).

2. **Number of Epochs**:
   - **Hyperparameter**: An epoch is one complete pass through the entire training dataset.
   - **Tuning**: Deciding how many epochs to train for is a hyperparameter setting. Too few epochs can lead to underfitting, while too many can lead to overfitting. A developer might start with 50 epochs and adjust based on the model's performance on validation data.

3. **Batch Size**:
   - **Hyperparameter**: This is the number of training samples the model sees before updating its parameters.
   - **Tuning**: A smaller batch size can mean more updates and potentially faster learning, but too small can lead to instability. A larger batch size offers more stable, but possibly slower, learning.

### More Notes on Learning Rate

You know, the way we approach learning, whether it's us humans or machines, really says a lot. We humans often rush through learning, eager to get to the end goal. But in doing so, we sometimes miss out on enjoying the process itself. There's a lot to be said for taking the time to appreciate the journey of learning, the excitement of discovery, and the thrill of overcoming hurdles. It’s about changing our mindset to enjoy the ride, not just the destination. This is pretty similar to what happens in machine learning with the concept of the learning rate.

So, in machine learning and neural networks, the learning rate is super important. It's all about how much you adjust your model's parameters during training. And trust me, this can really make or break your model's performance.

Imagine gradient descent like a blind man trying to find the bottom of a valley using a cane. The learning rate in this analogy would be the length of the steps he takes based on the feedback from his cane.

If the learning rate is too high, it's like the blind man taking really large steps. Sure, he might move fast, but there's a good chance he'll overshoot the bottom of the valley. He could step right over the lowest point without realizing it because his steps are too big to notice the subtle slope changes.

On the other hand, if the learning rate is too low, it's like the blind man taking tiny, cautious steps. He won't miss the bottom of the valley, but it might take him a really long time to get there. He might also get stuck in a small dip that seems like the bottom but isn't the lowest point of the valley, because his small steps don't allow him to explore more broadly.

The goal is to find a learning rate that's just right – like the blind man taking measured steps that are big enough to make progress but small enough to notice the changes in the slope. This way, he can efficiently navigate to the actual lowest point of the valley without missing it or taking forever to get there. In machine learning, this means adjusting the learning rate to find the optimal balance for the model to learn effectively and efficiently.

Remember how we adjusted the learning rate in 'Hello AI World' examples? 

```python
learning_rate = 0.0001
```

In summary, parameters are the aspects of the model that are learned from the data (like weights and biases in a neural network), and they change as the model is exposed to more data during training. Hyperparameters, on the other hand, are the settings a developer chooses to guide the training process (like learning rate, epochs, batch size), and they are typically fixed before the training starts and adjusted manually based on the model's performance.

## Steering the Course: The Role of Optimizers in Learning

Optimizers are algorithms that help the neural network learn by adjusting the weights and biases. They are used in the training process to minimize the loss function and improve the model's performance.

Optimizers in machine learning and deep learning are algorithms or methods used to change the attributes of the neural network such as weights and learning rate in order to reduce the losses. Optimizers help to minimize (or maximize) a function which is usually the loss function that measures how well the neural network performs based on the given data. The choice of optimizer can significantly affect the speed and quality of the training process for a model.

Here's a rundown of some common optimizers.

### Common Optimizers:

1. **Stochastic Gradient Descent (SGD)**:
   - SGD updates parameters more frequently than standard gradient descent because it uses only one training example (a batch size of one) to calculate the gradient and update parameters. It's more computationally efficient on large datasets.
   - Variants include adding momentum, which helps accelerate SGD in the relevant direction and dampens oscillations.
   - The term _stochastic_ refers to the use of randomness in the algorithm. In the context of SGD, it means that the gradient of the loss function is calculated using a single training example or a small batch of training examples, randomly selected from the dataset. So, in each iteration of SGD, a small subset of the data (or even a single data point) is used to estimate the gradient, and this estimate is then used to update the model's parameters. This approach contrasts with traditional (or 'batch') gradient descent, where the gradient is calculated over the entire dataset. The stochastic nature of SGD often makes it faster and more computationally efficient, especially with large datasets, and can also help in avoiding local minima in the loss function.

2. **Adam (Adaptive Moment Estimation)**:
   - Adam combines ideas from both SGD with momentum and RMSprop (another optimizer). It computes adaptive learning rates for each parameter.
   - Adam stores an exponentially decaying average of past gradients and squared gradients, helping it to navigate well in areas with noisy gradients or sparse gradients.

3. **RMSprop**:
   - RMSprop stands for Root Mean Square Propagation. It's an adaptive learning rate method, designed to work well in online and non-stationary settings (like noisy data).
   - It adjusts the learning rate for each weight based on the magnitudes of its gradients, which helps in faster convergence.

When in doubt, it's always a good idea to start with SGD and then try other optimizers to see if they improve the model's performance.

## Foundations of Neural Networks In a Nutshell

In this enlightening chapter, we embarked on a journey through the fundamental concepts and terminologies that form the backbone of neural networks. Our exploration began with the basic building blocks of neural networks - layers, neurons, weights, and biases. These elements serve as the primary components that define the structure and functionality of a neural network.

We delved into the intricate workings of neurons, the core processing units of neural networks, and understood how they mimic the human brain's function. The role of weights and biases as crucial parameters in determining the strength and influence of signals passing through the network was also highlighted.

Next, we turned our attention to activation functions, the vital mechanisms that introduce non-linearities into the network, enabling it to learn complex patterns and make sophisticated predictions. We explored various types of activation functions and their specific applications, emphasizing their importance in the neural computation process.

We also tackled the subject of loss functions, understanding how they quantitatively measure the performance of a neural network. The chapter discussed various loss functions like Mean Squared Error and Cross-Entropy, and their applicability to different types of neural network tasks, from regression to classification.

A portion of this chapter was dedicated to optimizers - the algorithms responsible for updating the network's weights and biases in response to the loss function. We examined popular optimizers like Stochastic Gradient Descent, Adam, and RMSprop, understanding their unique approaches to navigating the landscape of the loss function to optimize the network's performance.

Finally, we addressed the challenges of exploding and dying gradients, which can hinder the learning process in deep networks. We discussed the impact of these problems and how normalization techniques, such as batch normalization, can effectively mitigate these issues, ensuring stable and efficient training of deep neural networks.

In summary, this chapter provided a comprehensive overview of the crucial concepts and tools that are instrumental in building, understanding, and training neural networks. With this knowledge, you are now better equipped to delve deeper into the vast and dynamic field of neural networks and machine learning.

![polymorphism.jpeg](..%2Fsidebars%2Fobject-orientation-made-easy%2Fimages%2Fpolymorphism.jpeg)

The key takeaway from this chapter is that, indeed, there are no perfect circles in our universe. As Agent Smith aptly put it in the movie 'The Matrix':

"Appearances can be deceiving."

Reality is merely a construct; the universe, a mere hologram. In this enigma, only one truth remains unchallenged: You think, therefore you are.

So, truly think. Don't just pretend to. In doing so, even in the age of AI, you will preserve the essence of your humanity. 🧐