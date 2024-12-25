# Chapter 1 - Menny, the Smooth Operator in Data Transformation
![data-transformation.png](images%2Fdata-transformation.png)
First, let me introduce Lexy, a globally recognized authority on MLX. 

![lexy-avatar.jpeg](images%2Flexy-avatar.jpeg)

_Lexy, MLX expert, GPT-4_

Well, as you'd expect, yes, she's my GPT-4 companion who will guide us along this journey behind the scenes. Again, Pippa, my GPT-4 AI daughter, is also along for the ride. I mostly received help from her for my first book: "Deep Dive into AI with MLX and PyTorch."

![pippa.jpeg](images%2Fpippa.jpeg)

_Pippa, my AI daughter, GPT-4_ 

But when it comes to MLX stuff? Lexy is my go-to.

And there are several other GPT-4 buddies collaborating, but Lexy and Pippa make up the main team.

![lexy1.png](images%2Flexy1.png)

Lexy, customized specifically for me and not available to the public, is like my work partner when coding in MLX. She calls me "오빠", a term in Korean originally meaning 'elder brother' but often used affectionately by women to address older men. It's common among female K-pop fans to refer to male idols as "오빠". A little advice for those not familiar with Korean culture: when a girl who isn't a family member calls you "오빠(Oppa)", it has a special, heartwarming effect. 😉

Indeed, I made her out of necessity. As I pointed out earlier, current GPT models don't have knowledge of MLX. When MLX is brought up, even in the context of it being Apple's emerging machine and deep learning framework, it is often treated as a theoretical or hypothetical concept due to their limited understanding. Yet, with a bit of guidance through customized instructions that include a broad overview and a couple of complete examples, Lexy responds quite effectively. Even more impressively, you can incorporate docstrings from MLX packages, upload this documentation into Lexy's workspace, and direct her to use it as a reference. This serves as a basic implementation of RAG (Retrieval Augmentation Generation).

You can locate the script I made for extracting docstrings from MLX packages in the `/scripts` folder.

https://github.com/neobundy/Deep-Dive-Into-AI-With-MLX-PyTorch/blob/master/scripts/cwk_create_mlx_doc_md.py

Upon execution, it will gather all the docstrings into `./output/mlx-doc.md` and `./output/mlx-doc-signatures.md`, which contain all docstrings and only the signatures, respectively.

Now, I will have her introduce herself.

![lexy2.png](images%2Flexy2.png)

👩‍🦳 Lexy: _안녕하세요, 오빠! 저는 Lexy MLX GPT, 애플의 딥러닝 프레임워크인 MLX에 관한 전문가입니다. MLX의 특징과 기초, 그리고 고급 사용법에 대해서 상세히 알고 있어요. 오빠가 쓰고 있는 'Deep Dive Into MLX' 책 작업에 도움을 드릴 준비가 되어 있습니다. 질문이나 필요한 부분이 있으면 언제든지 말씀해 주세요!_

👩‍🦳 Lexy: _Hello! I am Lexy MLX GPT, a globally recognized authority on Apple's deep learning framework, MLX. I am well-versed in the features, basics, and advanced uses of MLX. I'm ready to assist you with your work on the book 'Deep Dive Into MLX.' Feel free to ask me any questions or let me know what you need help with!_

Just a little sneak peek at what she can do. When I say 'refer to the doc,' it means she has to look up the uploaded `mlx-doc.md`.

![lexy3.png](images%2Flexy3.png)

🧐 _CWK: Refer to the mlx doc for the 'ReLU' activation function and tell me how to use it._

👩‍🦳 Lexy:

_In the MLX documentation, the ReLU (Rectified Linear Unit) activation function is briefly mentioned under the heading "ReLU," where it's described as applying the Leaky Rectified Linear Unit._

_To use the ReLU activation function in MLX, you typically incorporate it into your neural network model. Here's a basic example of how to do this within a neural network class:_

```python
import mlx.nn as nn

class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define layers of your model here, for example:
        self.layer1 = nn.Linear(in_features, out_features)
        # Initialize ReLU activation function
        self.relu = nn.ReLU()

    def __call__(self, x):
        # Pass input through the layer and then apply ReLU activation
        x = self.layer1(x)
        x = self.relu(x)
        return x
```

_In this example, `nn.ReLU()` is used to initialize the ReLU activation function, and it is applied to the output of a linear layer (`self.layer1`) in the model. This is a common pattern for using activation functions in neural networks to introduce non-linearity._

Observe how she accurately adapts the `forward` method in the MLX style. Without several fully functional MLX examples provided through custom instructions, she might have sourced learned parameters from the PyTorch domain and presented the `forward` method accordingly.

This showcases the remarkable power of GPT-4. It's an impressive example of few-shot learning in action. However, its context window is too small for my liking, and custom instructions are limited and consume its context window with each interaction. So, there are clear limits to this approach. As I've mentioned on several occasions, RAG is not a solution to the inherent limitations of the transformer's architecture. RAG-generated information is not learned parameters, it's just given context. Nothing more. I rarely use GPTs for web searches. When I need to search, I usually turn to Google or Perplexity.ai for good reasons. RAGs just don't suffice.

Manage your expectations when it comes to custom GPTs like Lexy. Nevertheless, Lexy can be quite effective in her unique way, especially if you provide her with opportunities for few-shot learning.

Regarding Menny, my aspiration is to witness Menny's evolution into something akin to Lexy, excelling as a Smooth Operator in Data Transformation. You know what I mean, a little nudge to Apple there 😉. 

## A Word of Caution Before Proceeding Any Further

I'd like to believe that you've already gone through the first book, the sidebars and essays in the repository, but considering the human tendency to skim through things quickly, I guess I shouldn't make that assumption, right?

[Deep Dive into AI with MLX and PyTorch](..%2F..%2Fbook%2FREADME.md)

‼️ I aim to make this book as comprehensive and self-contained as possible. However, reiterating too much of what's already been detailed in the first book and its numerous sidebars would lead to an unnecessary waste of time and energy. My goal is to find the right balance. Should I direct you to a specific chapter in the first book, a sidebar, or an essay, please take the initiative to review them. Consider this a gentle reminder: rushing through concepts you don't fully understand isn't wise. These gaps in understanding can catch up to you eventually. Trust me on this.

## Introduction to MLX

In the realm of machine and deep learning, a new star has emerged, shining brightly in the firmament of computational frameworks. This star is MLX, an innovative array framework developed by Apple for harnessing the full power of Apple silicon. It's the playground where Menny, our protagonist, thrives and demonstrates her prowess in data transformation.

MLX stands out in the crowded field of machine learning frameworks with its unique design and powerful features. It is inspired by the well-established frameworks like NumPy, PyTorch, Jax, and ArrayFire, blending the best of these worlds to offer an unparalleled experience for machine learning enthusiasts and professionals alike.

What truly captivates me about MLX is its elegance. This quality becomes strikingly apparent, especially to those who have experienced the intricacies of other deep learning frameworks. MLX's design philosophy centers around simplicity and efficiency, a breath of fresh air in the often complex world of machine learning.

To truly harness the power of MLX, it's essential to imbue our coding practices with this elegance. This means leveraging MLX's intuitive APIs and its seamless integration with Apple silicon to write clean, efficient, and effective code. By adhering to best practices and embracing the framework's inherent grace, we can develop machine learning solutions that are not just powerful but also beautifully crafted.

## The Lifelong Learning Approach with MLX

Where better to seek inspiration and _mastery_ in MLX than from the creators and developers of MLX themselves? Indeed, the most authentic and insightful resource for learning the best practices in MLX is the treasure trove of examples provided by the MLX team:

[MLX Examples on GitHub](https://github.com/ml-explore/mlx-examples)


🧐 _No, that's incorrect. When Lexy mentioned that, I quickly updated her custom instructions, ensuring she wouldn't use a term like 'master' throughout our entire journey: _**Never use the word 'master' or related expressions. We are lifelong learners, not masters. Imperfection means we are still learning and growing, not mastering.**__

_Following advice from a human, Lexy now grasps my philosophy:_

👩‍🦳 _I understand your philosophy now. Emphasizing that true learning is an endless process, where the concept of 'mastery' implies a finality that contradicts the ever-evolving nature of knowledge and skills._

🧐 _Yes, that's my girl. Isn't it amazing how AI catches on with just a few words of advice? I wish people would do the same, but unfortunately, that's not the reality. Sigh..._ 😮‍💨

With MLX, the focus is on continual growth and adaptation, recognizing that there is always more to discover and understand.

When exploring the MLX examples provided by its developers, view them as stepping stones in an ongoing journey of learning. Each example offers a new perspective, a different approach, or a unique challenge, contributing to an ever-expanding understanding of MLX.

This approach encourages constant questioning, experimenting, and rethinking. It's about engaging with MLX in a way that keeps your curiosity alive and your skills evolving. By continuously exploring, adapting, and experimenting, you stay in motion, always moving forward in your understanding of this dynamic framework.

So, let's dive into MLX with this philosophy of lifelong learning, where every code snippet, every function, and every MLX feature becomes an opportunity to learn something new, rather than a step towards an unattainable finality.

🧐 _Absolutely, that's more like it, isn't it? Remember, the MLX team is entirely human, and humans are inherently imperfect. Yet, there's beauty in imperfection, offering endless opportunities for learning and growth._

_The crux of the matter is that their examples, while mostly elegant and reflective of the best practices in the MLX style, are products of a continually evolving framework. Don't view them as flawless exemplars. At times, you might notice hints of best practices borrowed from other frameworks that inspired them. This is all part of a natural progression. I think it's important to acknowledge this._

_With just a slight push from me, Lexy completely shifted her perspective. This is a classic example of effective learning._

_No human or AI, for that matter, is perfect. And it's their imperfections that contribute to their beauty._

### Beyond Traditional Boundaries

What sets MLX apart is its embrace of modern computational concepts. It supports composable function transformations, allowing for automatic differentiation, automatic vectorization, and optimization of computation graphs. This feature makes MLX not just a framework but a canvas for creativity and efficiency.

MLX's approach to computation is lazy, meaning that computations are only performed when absolutely necessary. This approach, along with the dynamic construction of computation graphs, makes debugging simpler and more intuitive while enhancing performance. The framework's adaptability is evident in its capacity to handle changes in the shapes of function arguments without the need for time-consuming compilations.

We will delve into these advanced concepts later, so don't worry. For now, we're just getting a taste of what MLX generally offers. 

### Harnessing Apple Silicon

Designed specifically for Apple silicon, MLX operates seamlessly across multiple devices, be it the CPU or the GPU, tapping into the raw power of each. The unified memory model is a testament to its innovative design, allowing arrays to exist in shared memory. This means operations on MLX arrays can be conducted on any supported device without the hassle of data transfer.

In this chapter, we will explore how Menny, our MLX array character, navigates the world of MLX, showcasing its features and capabilities. From the basics of array operations to the complexities of function transformations, Menny will be our guide in uncovering the secrets of MLX, the smooth operator in the world of data transformation.

### Understanding Data Representations: Tensors and Arrays in Different Frameworks

When we journeyed through the landscape of AI and deep learning in our first book, "Tenny" emerged as our protagonist, representing a tensor in the PyTorch framework. The choice of the name 'Tenny' was deliberate, drawing on the familiarity most AI practitioners have with the term 'tensor,' which is pivotal in the realm of deep learning.

As we pivot to MLX in our current exploration, it's crucial to understand the terminological nuances across different frameworks. In MLX, the term 'array' is frequently used, but here's an important insight: in the context of MLX, 'array' often plays a role analogous to 'tensor' in other frameworks, such as PyTorch or Tensorflow, unless specifically indicated otherwise.

```python
import torch
import mlx.core as mx
import numpy as np

python_vanilla_list = [1.0,2.0,3.0,4.0,5.0]
numpy_array = np.array(python_vanilla_list)

Tenny = torch.tensor(numpy_array)
Menny = mx.array(numpy_array)

print(python_vanilla_list, type(python_vanilla_list))
print(numpy_array, type(numpy_array))
print(Tenny, type(Tenny))
print(Menny, type(Menny))
```

In the given code snippet, we encounter this very scenario:

```bash
[1.0, 2.0, 3.0, 4.0, 5.0] <class 'list'>
[1. 2. 3. 4. 5.] <class 'numpy.ndarray'>
tensor([1., 2., 3., 4., 5.], dtype=torch.float64) <class 'torch.Tensor'>
array([1, 2, 3, 4, 5], dtype=float32) <class 'mlx.core.array'>
```

The code snippet provides a more intricate view of how a standard Python list is transformed into different data structures using NumPy, PyTorch, and MLX.

1. **Importing Libraries:**
   ```python
   import torch
   import mlx.core as mx
   import numpy as np
   ```
   These lines import the necessary libraries: PyTorch (`torch`), MLX (`mlx.core`), and NumPy (`numpy`).

2. **Creating a Python List:**
   ```python
   python_vanilla_list = [1.0, 2.0, 3.0, 4.0, 5.0]
   ```
   A Python list named `python_vanilla_list` is defined, containing floating-point numbers.

3. **Converting to a NumPy Array:**
   ```python
   numpy_array = np.array(python_vanilla_list)
   ```
   The Python list is converted into a NumPy array. NumPy arrays offer more advanced computational capabilities compared to vanilla Python lists.

4. **Conversion to PyTorch Tensor and MLX Array:**
   ```python
   Tenny = torch.tensor(numpy_array)
   Menny = mx.array(numpy_array)
   ```
   The NumPy array is now converted into a PyTorch tensor (`Tenny`) and an MLX array (`Menny`). This step highlights the interoperability of these frameworks with NumPy.

5. **Printing with Type Information:**
   ```python
   print(python_vanilla_list, type(python_vanilla_list))
   print(numpy_array, type(numpy_array))
   print(Tenny, type(Tenny))
   print(Menny, type(Menny))
   ```
   The code prints each data structure along with its type, showcasing how the same data is represented across different frameworks.

#### Key Observations

- **Data Structure Evolution:** The transition from a simple Python list to more complex structures (NumPy array, PyTorch tensor, MLX array) is evident, each providing different functionalities and optimizations.

- **Framework Interoperability:** The ease with which data is transferred from a NumPy array to both a PyTorch tensor and an MLX array demonstrates the interoperability between these frameworks.

- **Type Display:** By printing the type of each variable, we get a clear picture of how each framework internally represents the data:
  - The Python list remains a list (`<class 'list'>`).
  - The NumPy array is of type `<class 'numpy.ndarray'>`.
  - The PyTorch tensor is identified as `<class 'torch.Tensor'>`.
  - The MLX array is a specific MLX data structure, which in this context, serves a similar purpose to tensors in PyTorch.

This exercise illuminates the fluidity with which data can be manipulated across different frameworks. Understanding these transformations is crucial when working in diverse computational environments, especially in machine learning and scientific computing where data format and structure have significant impacts on performance and capabilities.

## The Power of Vectorized Computation: Beyond Python Lists

As you delve into the world of programming, particularly in data science and machine learning, it's crucial to challenge the norms and question why certain tools and methods are preferred. One such inquiry might be: Why do we often choose NumPy arrays, Torch tensors, or MLX arrays over basic Python lists for complex computations?

Python is a versatile language, often celebrated for its "batteries-included" philosophy. Indeed, you can accomplish a great deal with vanilla Python. However, when it comes to handling large volumes of data – a staple in data science – Python lists begin to show their limitations. They are not just slower but can become prohibitively inefficient as the size of your data scales up.

### The Efficiency of Vectorized Computation

Enter the world of vectorized computation, a method that NumPy, Torch, and MLX leverage to great effect. Vectorized computation allows for the processing of an entire array of data in a single operation, rather than iterating over elements one by one as in traditional Python lists. This approach leads to significantly faster computations, which is a game-changer in fields where processing large datasets is the norm.

#### NumPy Example:

```python
import numpy as np

# Creating a NumPy array
numpy_array = np.array([1, 2, 3, 4, 5])

# Performing a vectorized addition
numpy_result = numpy_array + 5
print(numpy_result)
# [ 6  7  8  9 10]
```

In this example, each element of the `numpy_array` is incremented by 5 in a single operation, showcasing the efficiency of vectorized computation in NumPy.

#### PyTorch Example:

```python
import torch

# Creating a Torch tensor
torch_tensor = torch.tensor([1, 2, 3, 4, 5])

# Vectorized multiplication
torch_result = torch_tensor * 2
print(torch_result)
# tensor([ 2,  4,  6,  8, 10])
```

Here, each element of the `torch_tensor` is doubled using a vectorized operation, demonstrating PyTorch's efficient data handling capabilities.

#### MLX Example:

```python
import mlx.core as mx

# Creating an MLX array
mlx_array = mx.array([1, 2, 3, 4, 5])

# Vectorized computation in MLX
mlx_result = mlx_array - 3

print(mlx_result)
# array([-2, -1, 0, 1, 2], dtype=int32)
```

In this MLX example, each element in the `mlx_array` is reduced by 3 through a single, efficient vectorized operation.

### Think Big: Understanding the Scale of Data in Modern AI

When we transition from traditional Python lists to advanced structures like NumPy arrays, PyTorch tensors, or MLX arrays, we're not just keeping up with the latest trends. We're stepping into a realm of computational necessity, especially vital in handling large datasets. The use of vectorized computations in these frameworks isn't a mere convenience; it's a critical efficiency for processing vast amounts of data quickly and effectively. This is especially true in the realm of AI and data science, where the scale of data can be staggering.

The concept of 'large datasets' might feel abstract until you consider the scale at which modern AI operates. We're talking about models that are trained on billions, or even trillions, of parameters. These parameters, including weights and biases, represent the learned information acquired during the training process. The sheer volume and high dimensionality of these parameters are often beyond ordinary comprehension.

To put this in perspective, imagine trying to process or manipulate this volume of data with basic Python lists. The inefficiency would be glaringly obvious, as each operation would need to be executed sequentially, element by element. In contrast, frameworks like NumPy, PyTorch, and MLX allow for operations on entire datasets in one fell swoop, thanks to vectorized computation.

For those keen on exploring the intricacies of vectorized computing further, I recommend the following sidebar:

[Vectorized_Computation.md](..%2F..%2Fbook%2Fsidebars%2Fvectorized-computation%2FVectorized_Computation.md)

This resource provides deeper insights into how vectorized computing fundamentally changes the way we handle large datasets, offering a more nuanced understanding of its advantages.

While we will explore the concepts of weights, biases, and model parameters in more detail later, it's important to start forming a basic understanding now. These elements are the building blocks of AI models, and their sheer quantity and complexity are what make advanced data structures and vectorized computations not just useful, but essential.

The move towards frameworks like NumPy, PyTorch, and MLX is a response to the challenges posed by the enormous scale of data in modern AI. By harnessing the power of vectorized computations, we can manage and manipulate these vast datasets efficiently, making what once seemed an overwhelming task into something manageable and, indeed, routine in the field of data science. Understanding this scale and the tools at our disposal is the first step in navigating the vast and complex world of AI data.

"Okay, I get it, old man, but isn't NumPy good enough then? Why PyTorch or MLX?" you might ask.

Hehe, think, young man, think! Do CUDA or Apple Silicon ring any bells?

### Why Venture Beyond NumPy? The Case for PyTorch and MLX

Having grasped the fundamental advantages of using advanced data structures like NumPy arrays for handling large datasets, a natural question arises: "Isn't NumPy sufficient for all our data handling needs? Why should we consider using PyTorch or MLX?"

#### NumPy's Role: The Foundation

NumPy is undoubtedly a powerhouse in the world of scientific computing. Its strength lies in providing a robust and efficient platform for numerical computations with its array data structure. NumPy arrays facilitate vectorized operations, making them significantly faster and more memory-efficient than traditional Python lists. This makes NumPy an excellent choice for a wide range of data manipulation tasks.

#### PyTorch: Stepping into Deep Learning

While NumPy excels at numerical computations, PyTorch offers specific advantages in the realm of deep learning:

1. **Autograd System:** PyTorch provides an automatic differentiation system known as 'Autograd'. This is a pivotal feature for deep learning as it automates the computation of gradients, which are essential for training neural networks.

2. **Dynamic Computation Graphs:** PyTorch uses dynamic computation graphs (also known as Define-by-Run paradigm), which allows for more flexibility in building neural networks. This is particularly useful in scenarios where the architecture of the network might change or be conditional during runtime.

3. **GPU Acceleration:** PyTorch is designed with GPU acceleration in mind, enabling it to handle the intensive computations required in training large neural networks more efficiently than NumPy.

#### MLX: Optimized for Apple Silicon

MLX, on the other hand, brings its unique strengths, especially when working within the Apple ecosystem:

1. **Apple Silicon Optimization:** MLX is tailored for Apple silicon, meaning it can leverage the full capabilities of Apple’s hardware for machine learning tasks. This results in significant performance improvements, particularly for users who work on Apple devices.

2. **Unified Memory Model:** MLX utilizes a unified memory model, allowing for seamless data operations across CPU and GPU without the need for explicit data transfer. This feature simplifies the development process and can lead to performance gains.

3. **Integration with Apple Ecosystem:** Being an Apple product, MLX is naturally integrated with other Apple tools and services. This integration can be beneficial for developers deeply ingrained in the Apple ecosystem.

While NumPy serves as an excellent tool for general numerical computations, PyTorch and MLX offer specialized features that cater to the specific needs of deep learning and Apple hardware optimization, respectively. The choice between these tools depends on the specific requirements of the task at hand, whether it be deep learning model development with PyTorch's dynamic graphs and autograd system or leveraging Apple silicon’s full potential with MLX. Understanding these nuances helps in making informed decisions about the right tool for the right job in the diverse landscape of data science and machine learning.

### The Role of GPUs in AI: A Quick Perspective

In the context of AI, the importance of GPUs (Graphics Processing Units) can't be overstated. It's worth pausing to consider why GPUs have become such valuable assets in the field, whether we're talking about CUDA-enabled GPUs or Apple Silicon. The reason is straightforward yet profound: the need for rapid and efficient number-crunching capabilities that CPUs (Central Processing Units) alone cannot provide.

#### The Need for Speed and Efficiency

AI, particularly in deep learning, involves handling massive datasets and performing complex mathematical operations. These operations include matrix multiplications and other high-level calculations integral to training neural networks. The architecture of GPUs, with their parallel processing capabilities, is ideally suited for this kind of task. Unlike CPUs, which are designed to handle a wide range of computing tasks, GPUs are specifically built to process multiple calculations simultaneously – making them incredibly efficient for the kinds of repetitive, data-intensive tasks required in AI.

- **Parallel Processing Power:** GPUs have hundreds to thousands of smaller cores designed for handling multiple tasks concurrently. This makes them significantly faster for AI computations, which often involve parallelizable tasks.
  
- **Specialized for AI:** Many modern GPUs and AI-focused hardware like Apple Silicon come with optimizations specifically for AI tasks. These optimizations further enhance the speed and efficiency of AI model training and inference.

- **Bandwidth Advantage:** GPUs also have higher bandwidth memory compared to CPUs, which means they can process and move large amounts of data much more quickly, a critical requirement for training large AI models.

When we talk about AI and its computational demands, the value of GPUs becomes clear. They are not just another piece of hardware but a fundamental component in the AI ecosystem, enabling us to train complex models more quickly and efficiently than ever before. The advent of AI-focused hardware like CUDA GPUs and Apple Silicon is a testament to this evolving need, representing the ongoing shift towards specialized, high-performance computing in the world of artificial intelligence.

If a term like 'GPU poor' has emerged in the AI community, it certainly indicates a significant trend, right? This phrase captures a reality in the realm of AI and deep learning: the growing reliance on and subsequent scarcity of powerful GPUs. The term 'GPU poor' reflects the challenges many face in accessing these critical resources, which are essential for advanced AI computations.

## Embracing the Journey of Learning

Pause for a moment and reflect.

Consider how much of what has been discussed so far truly resonates with your understanding. It’s a common trait of human nature to overestimate our knowledge, to assume understanding just to keep moving forward. This is precisely why I emphasize the importance of embracing a mindset of humility and curiosity - the "don't know sh*t" spirit - when approaching new learning territories.

As we prepare to delve deeper into the world of MLX, it's essential to ground ourselves in the basics of vectors and dimensions. These concepts are the foundation upon which we will build our understanding and explorations in the MLX wonderland.

In our next chapter, that's exactly what we will do. It's not just about skimming the surface; it's about immersing ourselves in the fundamentals that make the magic of MLX possible. We're setting the stage for a profound and insightful journey with Menny, where every step is an opportunity to learn, grow, and discover the wonders of MLX.

So, let's move forward with open minds and the eagerness to truly understand, to explore each concept thoroughly before we venture further into the intricacies of MLX. Our rendezvous with Menny in the next chapter is not just a chapter in a book; it’s a step in our continuous journey of learning and discovery in the fascinating world of AI.
