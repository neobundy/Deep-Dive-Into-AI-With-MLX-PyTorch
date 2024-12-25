# 'mlx.nn' is a sub-library of MLX, which contains classes to build neural networks. 'nn' stands for neural network.
# 'mlx.optimizers' is a sub-library of MLX, which contains classes to optimize neural networks.
# Note these importing conventions. You can use different import conventions, but this is the most common one.
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

TITLE = "Hello AI World - MLX"

print(TITLE)
print("Let's train a simple AI model using MLX.")

# Create a dataset of twenty random integers as input (x values).
x_train = [40, 1, 16, 95, 79, 96, 9, 35, 37, 63, 45, 98, 75, 48, 25, 90, 27, 71, 35, 32]
print("Training dataset x values: ", x_train)
input("Press enter to continue...")

# Calculate the corresponding output (y values) based on a predefined linear relationship: y = 10x + 3.
# y values are the labels for the training data, which means each y value is the correct answer for the corresponding x value. If 10 is the x value, then 103 is the y value.
# The model will be trained to learn this relationship. It will learn that the output is 10 times the input plus 3.
y_train = [403, 13, 163, 953, 793, 963, 93, 353, 373, 633, 453, 983, 753, 483, 253, 903, 273, 713, 353, 323]

# Please get used to the term 'weights' and 'biases'. They are the parameters of the model. They are the variables that the model learns during training. In the equation '10 * x + 3', 10 is the weight and 3 is the bias.
# You'll see tons of these examples in the AI equations. Also note that in this case, single weight and bias are used. In more complex models, there will be multiple weights and biases.
# The most confusing part is that the weights and biases are also called parameters. So, parameters are the variables that the model learns during training. They are the weights and biases.
# Furthermore, weights and biases are stored in tensors. So, parameters are also tensors. Again, linear algebra comes into play here.
# Weights and biases are mathematically manipulated as matrices and vectors. Why do we need GPUs for AI? Because GPUs are optimized for matrix and vector operations. GPUs are much faster than CPUs for matrix and vector operations. That's why GPUs are used for AI.


print("Training dataset y(label) values: ", y_train)
input("Press enter to continue...")

# Convert the training data lists to tensors that MLX can work with.
# Note that tensors are basically multi-dimensional arrays. They are the fundamental data structure in MLX.
# Conventionally, PyTorch calls them tensors, while MLX calls them arrays. They are the same thing, but in the AI world, the term tensor is more commonly used.
# You might think multi-dimensional arrays are just matrices. That's true, but tensors can have more than two dimensions, even hundreds or thousands of dimensions.
# You won't be able to visualize tensors with more than three dimensions, but complex AI models can have tensors with hundreds or thousands of dimensions.
# The following code converts the training data lists to tensors and reshapes them to the correct dimensions.
# The -1 in the reshape function means that the number of rows is unknown. The reshape function will automatically calculate the number of rows based on the number of columns.
# For example, if the input is a list of 20 numbers, the reshape function will reshape it to a 20x1 tensor.
# Ah, dimensions, a constant source of confusion for beginners. Don't worry, you'll get used to it.
# Just remember the x_train and y_train lists are 1-dimensional lists. The reshape function converts them to 2-dimensional tensors.
# The first dimension is the number of rows, which is 20 in this case. The second dimension is the number of columns, which is 1 in this case.
# Why the conversion from 1d to 2d? Because MLX expects the input to be 2d tensors. It's just a convention. You'll get used to it.
x_train_tensor = mx.array(x_train, dtype=mx.float32).reshape(-1, 1)
y_train_tensor = mx.array(y_train, dtype=mx.float32).reshape(-1, 1)

# Define a simple model with just one input and one output (linear regression).
# Also note that different layers expect different input shapes. For example, the linear layer expects a 2d tensor as input.
# Note that PyTorch used in_features and out_features instead of input_dims and output_dims. Just a naming difference.
model = nn.Linear(input_dims=1, output_dims=1)
mx.eval(model.parameters())

# Set the loss function and the optimizer to update the model.
# Learning rate set to 0.0001. Try changing this value to see how it affects the training process.
# Learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
# Parameters are mostly learnable while hyperparameters are mostly tunable.
# - Parameters are the internal variables of the model that are learned from the training data. For example, in a linear regression model, the weights and biases are parameters.
# - Hyperparameters, on the other hand, are the settings of the training process that are set before training starts and are not learned from the data. Examples of hyperparameters include the learning rate, the number of epochs (iterations over the entire dataset), and the batch size.
#
# So, parameters are "learnable" because they are learned from the data during training, while hyperparameters are "tunable" because they are manually set and adjusted by the developer to optimize the learning process and the performance of the model.
learning_rate = 0.0001

# The loss function measures the difference between the predicted output and the expected output.
# There are many different types of loss functions. The mean squared error (MSE) is a common one.
# Convert the loss function to a function that computes both the value and gradients
# MLX is different in how gradients are computed. Unlike PyTorch, where you typically compute the gradients by calling the backward() method on a loss tensor, MLX uses a different approach that is similar to JAX.
# In MLX, you use the value_and_grad function to create a new function that, when called, computes both the value of the original function and its gradient with respect to its parameters. This is what is meant by "composable function transformations".
# In this line, nn.value_and_grad(model, loss_fn) returns a new function loss_and_grad_fn. When you call this function with some inputs, it computes both the value of loss_fn and its gradients with respect to the parameters of model.
# This approach is more functional and composable, as it allows you to create complex functions by combining simpler ones, and automatically computes gradients for you.
# The term "composable" in this context refers to the ability to combine simple functions to create more complex ones. This is a key feature of functional programming.  In the context of MLX and similar libraries like JAX, "composable function transformations" means that you can take a function that computes some value (like a loss function), and easily create a new function that computes both the value and its gradients. This is done using the value_and_grad function.
# This approach allows you to build complex computations by combining simple, reusable pieces, and the library automatically handles the computation of gradients for you. This makes it easier to build and reason about complex models and algorithms, especially in the context of machine learning and optimization where gradient computations are a key part.


def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# At first, I directly ported the PyTorch example unknowingly. It didn't work as intended. I got nan values for the loss. It was due to my lack of understanding of MLX on composable function transformations. I had to ask Awni Hannun (Machine Learning Research @Apple) for help. https://x.com/awnihannun

# The optimizer updates the model's parameters by using the gradients computed by the loss function.
# Gradients are the partial derivatives of the loss with respect to the model weights. They point in the direction of steepest ascent, which is the direction to move to minimize the loss.
# There are many different types of optimizers. Stochastic gradient descent (SGD) is a common one.
optimizer = optim.SGD(learning_rate=learning_rate)

# Define the number of training iterations known as epochs.
# In each epoch, the model is trained once on the entire set of training data.
# Monitoring the loss after each epoch helps track the model's improvement.
# Adjust the number of epochs to influence the training outcome: too few may lead to underfitting where the model fails to capture the underlying trend, while too many can cause overfitting where the model memorizes the training data rather than learning to generalize, potentially decreasing its real-world accuracy.
num_epochs = 5000

input(f"The model will be trained for {num_epochs} epochs. Press enter to continue...")

for epoch in range(num_epochs):
    # Generate model predictions through a forward pass.
    # The forward pass computes the model's output based on the current input and parameter values (weights and biases).
    # Minimizing the loss, which assesses the discrepancy between predictions and true values, is the goal of training.
    # Calculating the loss is a critical step that precedes the computation of gradients.
    # Gaining an understanding of the forward pass and gradient computation is essential; these concepts are cornerstones in neural network training and are discussed frequently in literature.
    # During training, the model's weights and biases are iteratively tuned to reduce the loss, thereby polishing the model's performance.
    # This process relies on linear algebra—weights and biases are mathematically manipulated as matrices and vectors.
    loss, grads = loss_and_grad_fn(model, x_train_tensor, y_train_tensor)
    # Refine the model through a process called optimization.
    # The function loss_and_grad_fn computes gradients, which measure how much the loss changes with respect to each weight and bias in the model.
    # These gradients are used by the optimizer to update the model's parameters, aiming to minimize the loss.
    # The learning rate is a crucial factor that determines the size of the updates to the parameters; it influences how quickly the model learns.
    # The loss function evaluates the model's predictions against the true values, guiding the optimization process.
    # Through iterative adjustments based on calculated gradients, the model becomes more accurate.
    # Understanding the principles of gradient computation and optimization is essential—they're the backbone of neural network training.
    # A nod to calculus and its founders, Newton and Leibniz, for their contributions that underpin these concepts.
    optimizer.update(model, grads)
    # Compute the new parameters but also the optimizer state. This is typically done after an update step during training, to ensure that the parameters and the optimizer's state are correctly updated for the next iteration.
    mx.eval(model.parameters(), optimizer.state)
    # Print the loss every 10 epochs to see how the prediction error is improving(loss is decreasing).
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.5f}')

# Test the model by predicting the output for a given input, in this case, x = 100.
# If training is successful, the model should return a value close to 1003.
input("Training complete. The model will now predict the output for x=100. Press enter to continue...")

input_number = mx.array([100.0]).reshape(-1, 1)
predicted_output = model(input_number)

food_for_thought = """
The equation is y = 10x + 3.
So for x = 100, y should be 10 * 100 + 3, which is 1003.
You might notice that the predicted output is slightly different. Consider why this might be the case.
Hint: The model isn't aware of the exact equation. It works to uncover the underlying relationship between x and y through training on the data. 
It doesn't have prior knowledge of the equation itself. This is why the predicted value might be slightly off and vary each time.

To ensure the effectiveness of the training process and the accuracy of the resulting model, the following factors are essential:

- A sizeable training dataset is required to provide the model with enough information.
- The training dataset must accurately reflect the real-world correlation between x and y to ensure effective learning.
- An appropriate number of training iterations, known as epochs, is necessary for the model to learn from the data adequately.
- The learning rate needs to be chosen carefully; it influences the size of steps the model takes during optimization.
- The structure of the neural network—its architecture—must align with the complexity and type of the data it is learning from.

Selecting the correct hyperparameters, such as the learning rate and the number of epochs, is crucial to a well-trained model. Inappropriate values might result in training issues. 
For instance, encountering 'nan' values during training signals a problem—'nan' stands for 'not a number' and indicates that the model's learning process has gone astray, often due to numerical errors or instability.
"""

print("--" * 20)
print(f'For input x = 100, the predicted y value is: {predicted_output.item()}')
print("--" * 20)

print(food_for_thought)
