# Part I - Culinary Secrets of the AI Kitchen

Welcome to the exhilarating world of Artificial Intelligence, a domain where technology meets creativity, and the boundaries of what's possible are constantly being redefined. Part I of this book is designed to be your gateway into this fascinating world, offering a comprehensive yet accessible exploration of AI's fundamental concepts and applications. Whether you're a budding enthusiast, a seasoned professional looking to update your knowledge, or simply curious about how AI is reshaping our world, this section is tailored to provide you with a solid foundation. Here, we'll embark on a journey through the basics of machine learning, delve into the intricate workings of algorithms, and explore how these digital marvels are applied in various sectors to solve real-world problems.

As we progress, you'll find that each chapter is crafted not just to impart knowledge, but to ignite a passion for learning and innovation. We start by demystifying the core principles of AI, breaking down complex ideas into understandable segments. You'll get acquainted with the key terms and concepts that form the language of this field. This part of the book aims to give you a panoramic view of the AI landscape. It's an exciting journey that promises to equip you with the tools and insights needed to appreciate and engage with AI in a meaningful way.

As we delve deeper into the practical aspects of AI, Part I introduces you to Tenny, a virtual AI analyst designed to demonstrate the application of AI in the financial sector. Tenny is not just a theoretical construct but a fully functional stock price prediction model, embodying the principles and techniques you will learn throughout this section. Through Tenny, you'll witness the power of AI in analyzing and forecasting market trends, a task that encapsulates large volumes of data and complex predictive analytics. This example serves as a real-world application of AI, grounding abstract concepts in a tangible and relatable context. By building and refining Tenny, you will gain hands-on experience in data processing, model training, and the nuances of machine learning algorithms, all aimed at predicting stock prices with increasing accuracy. Tenny's journey through the labyrinth of financial data is not just an academic exercise; it exemplifies the practical utility and transformative potential of AI in one of the most dynamic and impactful realms of the modern world.

# Prolog - Welcome to "Hello AI World"

Deep Learning is a subset of Machine Learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—in order to 'learn' from large amounts of data. While a neural network with a single layer can still make approximate predictions as in the simple AI hello world example, additional hidden layers can help optimize the accuracy.

In this `hello-ai-world` example, I'm using a simple form of a neural network model known as linear regression. The goal of this model is to find the best linear relationship between the input variable `x` and the output variable `y`. This is done by finding the best values for `weight` and `bias` in the equation `y = weight * x + bias`, which represents a line in two-dimensional space.

The model is trained on a dataset of 20 random integers (the `x_train` values), and their corresponding `y_train` values are calculated using the equation `y = 10x + 3`. The model's task is to learn this relationship between `x` and `y` from the training data.

The training process involves feeding the `x_train` values into the model, calculating the difference between the model's predictions and the actual `y_train` values (the loss), and adjusting the model's parameters to minimize this loss. This process is repeated for a specified number of iterations (epochs).

After training, the model can predict the `y` value for a given `x` value. In this example, we test the model by asking it to predict the output for `x = 100`.

This is a basic introduction to deep learning. As you delve deeper, you'll encounter more complex models and techniques, such as convolutional neural networks (CNNs) for image processing, recurrent neural networks (RNNs) for time series analysis, and many others. Interested in Stable Diffusion? Lots of CNNs in there.

You'll also learn about techniques to optimize the training process, such as different types of loss functions, optimizers, and regularization methods.

Detailed explanations need code by the side. Code examples are heavily commented in simple terms. Please read them like a book. Don't just skim through them. Type them out yourself. Run them. Play with them. Break them. Fix them. Make them your own. That's the best way to learn.

At first, you won't be able to understand everything. That's okay. Just keep reading and practicing. You'll get there.

Both PyTorch and MLX examples are provided. They do the same thing. They only differ in syntax and structure. 

Frameworks and libraries in the context of AI are akin to toolboxes, providing a range of pre-built functions and structures that simplify the process of building AI models. They also serve as valuable learning resources for understanding how these models function. PyTorch, developed by Meta (formerly Facebook), is a mature and widely-used framework in the field. On the other hand, MLX is a newer framework developed by Apple. Despite its recent introduction and relatively small user base, MLX is a promising tool for learning AI, particularly for those utilizing Apple's silicon hardware.

The `hello-ai-world` example is demonstrated using PyTorch and MLX. Even though the core code is functionally the same for both, the syntax and logic execution methods are customized to fit the unique needs of each framework. The PyTorch version is provided as a reference for those who are already familiar with the framework. The MLX version is the primary example for this project.

## AI World Simplified

Don't get caught up in the details just yet. 

AI is a vast field. It's easy to get lost in the jargon and technical terms. It's also easy to get overwhelmed by the sheer volume of information.

As of now, just read the code and comments and familiarize yourself with the concepts. You'll get the gist of it. If you're new to coding, don't worry. You'll get used to it. Just keep reading and typing out the code. Remember the first goal is reading comprehension. You can't write code if you can't read it.

Note that in this example only the training set is used. In a real-world scenario, the dataset would be split into three sets: training, validation, and test. The model would be trained on the training set, and the validation set would be used to evaluate the model's performance during training. The test set would be used to evaluate the model's performance after training.

Following examples will cover these topics in detail.

When you look at the training set(x_train and y_train), you'll notice that the y_train values are calculated using the equation y = 10x + 3. The model is trained on the x_train values and their corresponding y_train values. The model's task is to learn this relationship between x and y from the training data.

The training process involves feeding the x_train values into the model, calculating the difference between the model's predictions and the actual y_train values (the loss), and adjusting the model's parameters to minimize this loss. This process is repeated for a specified number of iterations (epochs).

After training, the model can predict the y value for a given x value. In this example, we test the model by asking it to predict the output for x = 100.

The simplest workflow of AI is training and inference. Training is the process of feeding data into a model and adjusting its parameters to minimize the loss. Inference is the process of using the trained model to make predictions on new data.

In order to train a model, you need a dataset. A dataset is a collection of data points. Each data point consists of one or more features and a label. The features are the inputs to the model, and the label is the output that the model is trying to predict. In the examples, the features are the x values, and the labels are the y values. In some advanced cases, the features and labels can be images, audio, or text. Even without labels, you can train a model to find patterns in the data. In these cases, the model is said to be unsupervised. In the examples, the model is supervised because it is trained on labeled data.

The provided example is a typical machine learning workflow rather than a deep learning workflow. In a deep learning workflow, the model is trained on a large dataset, and the training process is repeated for many epochs. In this example, the model is trained on a small dataset for a single epoch. This is done for simplicity and to make the example run faster. 

Take GPTs or Stable Diffusion, for instance. They're deep learning models. They're trained on large datasets for many epochs. They take a long time to train. They're also very complex. Hopefully, we'll get there. For now, let's keep it simple.

In this example, we don't save the model after training. In a real-world scenario, you would save the model so that you can use it later for inference. Any model can be saved after training and loaded for inference. 

Consider why they're called model 'checkpoints'. It's similar to game checkpoints. You save your progress and can load it later. These models aren't perfect; they're just advanced enough to progress you to the next stage. You can always revisit and further train them. Like in a game, you can go back to a previous checkpoint or any other saved checkpoints.
Every AI model is saved when it's deemed sufficiently good. Basically, it's a checkpoint🤗

## Embracing AI-Assisted Learning

Python is a highly favored language in the AI sphere for its simplicity and versatility, complemented by a vast and active community. It's also a great language for beginners to learn. However, the sheer volume of Python libraries and frameworks can be overwhelming for newcomers.

When tackling unfamiliar code, lean on AI-assisted tools such as GPT. These sophisticated assistants are invaluable educational resources. By posing queries to a GPT model—GPT-4 recommended—you'll receive clarifications that can deepen your understanding significantly. Engage in a continuous, informative dialogue to enhance your learning journey step by step.

In IDEs like VSCode or PyCharm, integrate AI-powered plugins such as GitHub Copilot. These context-sensitive tools are designed not just to write code but also to facilitate your understanding of it. They act as both a coding companion and a real-time learning assistant. Embrace these innovations to make coding more approachable and your learning curve less steep.

## What's Next?

If you're just diving into AI, you're probably asking yourself:

- What exactly are vectors, matrices, or tensors, and why are they important?
- Why can't we just stick to using basic Python lists or arrays?
- What's the deal with these darn dimensions and reshaping?

These questions are crucial for understanding AI basics, and you'll need to know them as you move forward.

For me, dimensions are the trickiest part; they're likely to be a constant challenge in your AI journey. Make sure to understand them well.

No worries, though. I'm going to delve into all these topics in the next examples. Just keep on reading and practicing for now. You're on the right track.

The 'hello AI world' example is really just there to spark your interest.