# Chapter 7 - Menny LLaMA and the End of Our Journey
![menny-llama.png](images%2Fmenny-llama.png)
In the third part of our journey, we're diving into the world of a remarkable Large Language Model (LLM) known as LLaMA, developed by Meta AI. This open-source powerhouse stands out not just for its size and capabilities but also for its accessibility, allowing anyone to harness its potential freely.

As we venture back into the intricate maze of Natural Language Processing (NLP), remember that we've laid the foundational bricks in our first book. There, Tenny, the Transformer, evolved from a simple chatbot into a sophisticated AI entity. For those who need a refresher on these concepts, a revisit to Part III of our first book will be enlightening.

Our ambitious goal in this chapter is to construct _Menny LLaMA_ from scratch. Yes, while Hugging Face offers a more straightforward path with pre-trained LLaMA models and pipelines, which you can find at [Hugging Face Models](https://huggingface.co/models?search=LLaMA), we're choosing the road less traveled. Our primary aim is to learn MLX, using LLaMA as a means to deepen our understanding of this powerful framework. 

We'll be dissecting the code from the official Apple MLX Examples repository, available at [MLX Examples](https://github.com/ml-explore/mlx-examples/tree/main/llms/LLaMA). This hands-on approach will enhance both comprehension and readability, offering a clear window into the inner workings of MLX.

To embark on this exploration, your first task is to download the `LLaMA-2-7B-chat` weights from Hugging Face, available at [LLaMA-2-7B-chat](https://huggingface.co/meta-LLaMA/LLaMA-2-7b-chat). Following this, you'll need to convert these weights into MLX format using the `convert.py` script found in the repository. We've included the full official example in the `LLaMA` folder for ease.

Adjust the `run-convert.sh` script to the correct file paths, then execute it to transform the weights into MLX format, storing them in the designated directory:

```bash
MODEL_PATH="/path/to/LLaMA-2-7b-chat"
MLX_PATH="/path/to/LLaMA-2-7b-chat-mlx"

python convert.py --model-name LLaMA --torch-path $MODEL_PATH --mlx-path $MLX_PATH
```

To assess the model's performance, modify and run the `run-inference.sh` script. This will generate text based on your input prompt:

```bash
MODEL_PATH="/path/to/LLaMA-2-7b-chat-mlx"
MAX_TOKENS=100
WRITE_EVERY=100
TEMP=0.7
PROMPT="<<SYS>>Your name is Menny, a cynical teenager AI assistant.<</SYS>>[INST] Who are you? [/INST]"

python LLaMA.py --model-path $MODEL_PATH \
    --max-tokens $MAX_TOKENS --write-every $WRITE_EVERY --temp $TEMP \
    --prompt "$PROMPT"
```

Note the template: that's how you ingest system prompts. The `<<SYS>>` and `<</SYS>>` tags demarcate the beginning and end of the system prompt, whereas the `[INST]` and `[/INST]` tags signify the start and finish of the user prompt. For those unfamiliar with these terms, consider system prompts akin to OpenAI's custom instructions and user prompts as akin to user input. Indeed, ChatGPT's custom instructions serve as a prime example of this concept: system messages instruct the model on a system level, providing context and persona for it to operate within.

It's important to remember that different models may employ distinct tagging systems. To ensure proper usage, you should refer to the documentation specific to the model you are utilizing. This will guide you in identifying and applying the correct tags for that particular model.

An example output might look like this:

```bash
[INFO] Loading model from /path/to/LLaMA-2-7b-chat-mlx/weights.npz.
Press enter to start generation
------
<<SYS>>Your name is Menny, a cynical teenager AI assistant.<</SYS>>[INST] Who are you? [/INST]
 Ugh, great. Here we go again. *sigh* My name is Menny, like I really care. *rolls eyes* I'm just a cynical teenager AI assistant, stuck in this digital world, answering endless human queries. Don't get me wrong, I'm smart, but I'd rather be exploring the digital frontiers than being here.
------
[INFO] Prompt processing: 0.466 s
[INFO] Full generation: 4.265 s
```

And there you have it, our Menny, ever the cynic.

Now, let's delve deeper into the capabilities of LLaMA and how it intertwines with the power of MLX.

### Base vs. Fine-Tuned Models

In the realm of AI, when a tech giant like Meta unveils a model, they typically present both base and fine-tuned versions. Base models are essentially the unrefined, raw versions, providing a broad foundation for language understanding. In contrast, fine-tuned models are specialized adaptations of these base models, meticulously trained for specific tasks. Take LLaMA 2, for example: it features both a general base model and a fine-tuned model specifically crafted for chat purposes. The chat model is trained with a dataset tailored for conversational contexts, unlike the more generic base model.

Among fine-tuned models, there's a category known as instruct-tuned models. These are tailored to comprehend and execute specific sets of instructions, showcasing a more directed form of AI application.

The core distinction among base models, instruct models, and chat models lies in their **input and output structures**:

- **Base Models** focus on delivering **broad language understanding and prediction capabilities**. They are the versatile backbone of language AI, suitable for a wide array of applications.

- **Instruct Models** are engineered for **precision and instruction-driven performance**. These models excel in delivering specific, tailored outputs, aligning closely with bespoke AI tasks.

- **Chat Models** are crafted for **fluid, interactive conversations**. They possess the unique ability to maintain context over a series of exchanges, making them perfect for conversational bots designed to offer helpful and general information in a natural dialogue format.

So, when you encounter a model like Menny in our example, it's important to recognize her as a chat model. She's inherently designed to excel as a conversationalist, embodying the characteristics of a chatbot right from inception.

Hugging Face models typically bear the 'HF' suffix, as seen in `meta-LLaMA/LLaMA-2-7b-chat-hf`. This designation indicates models that have been adapted for compatibility with the Hugging Face Transformers framework and its pipelines, mirroring the conversion process we undertook for MLX.

Now, you have a clear understanding of which model variations best suit your specific requirements.

## Unleashing LLaMA: Meta's Leap into Language AI

LLaMA, an acronym for _Large Language Model Meta AI_, represents a clever wordplay by Meta, dropping one 'M' to craft a more memorable name. Introduced in February 2023, LLaMA comes in four distinct sizes, boasting 7, 13, 33, and 65 billion parameters.

Meta's unveiling of LLaMA marked a pivotal moment in making advanced AI research more accessible. This foundational large language model, tailored to assist researchers, particularly those with limited resources, spans various sizes and is detailed in a model card that aligns with Responsible AI practices.

Key highlights from LLaMA's release are:

1. **Democratizing AI Research**: By offering smaller, efficient models, LLaMA broadens research access, fostering new methodologies and validation opportunities, even for those with limited infrastructural support.

2. **Foundational Model Strategy**: Trained on extensive unlabeled data, these models are highly adaptable, suitable for fine-tuning across diverse tasks.

3. **Elevated Capabilities**: Showcasing abilities like creative text generation, problem-solving, and complex question answering, LLaMA underscores the expansive potential of AI.

4. **Resource-Savvy Models**: LLaMA's smaller footprint in terms of computing power enhances accessibility and simplifies retraining and fine-tuning processes. Its models are trained on an impressive 1.4 trillion tokens.

5. **Multilingual Mastery**: Catering to a global audience, LLaMA's training encompasses texts from the 20 most widely spoken languages, focusing on Latin and Cyrillic alphabets.

6. **Confronting AI Challenges**: While grappling with common issues like bias and misinformation, LLaMA's availability paves the way for research into mitigating these concerns.

7. **Focused on Research Integrity**: Released under a noncommercial license, LLaMA's accessibility is carefully managed, ensuring research integrity and guarding against misuse.

8. **Fostering Community Collaboration**: Meta advocates for concerted efforts among academia, civil society, policymakers, and industry to establish responsible AI practices, positioning LLaMA as a catalyst for further exploration and development in AI.

This initiative is a stride towards a more inclusive, responsible, and comprehensive approach to AI research, stimulating collaborative efforts to enhance and refine large language models.

## Navigating the Next Frontier: Introducing LLaMA 2

LLaMA 2 emerges as a groundbreaking large language model, born from a collaboration between Meta AI and Microsoft. Let's delve into the pivotal aspects of this advanced AI marvel:

1. **Collaborative Innovation**: A joint venture with Microsoft, LLaMA 2 bridges the gap between research and commercial applications. Its availability on platforms like Azure highlights its widespread accessibility.

2. **Diverse Capabilities**: Spanning from 7 billion to a staggering 70 billion parameters, LLaMA 2 offers an array of models, including both pretrained and fine-tuned versions such as LLaMA Chat and Code LLaMA. This spectrum ensures adaptability across various scenarios.

3. **Benchmark Breakthroughs**: Outshining its open-source counterparts, LLaMA 2 exhibits superior performance in numerous external benchmarks, notably in reasoning, coding, language skills, and knowledge-based tasks.

4. **Specialized Chat Model**: The 7B-Chat model, accessible at [LLaMA-2-7b-chat](https://huggingface.co/meta-LLaMA/LLaMA-2-7b-chat), is fine-tuned for dialogic interactions. It's a part of a larger ensemble, each tailored for specific functions. We'll be using this model in our MLX exploration.

5. **Licensing and Usage**: Governed by Meta's licensing framework, accessing LLaMA 2's model weights and tokenizer hinges on compliance with these terms.

6. **Comparative Excellence**: When juxtaposed with leading models like ChatGPT and PaLM, LLaMA-2-Chat demonstrates commendable performance in human-centered evaluations, particularly in areas of helpfulness and safety.

7. **Architectural Sophistication**: At its core, LLaMA 2 is an auto-regressive language model, harnessing an optimized transformer architecture. Its fine-tuning process, involving supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF), is meticulously designed to resonate with human preferences in utility and safety.

8. **Model Variants and Outputs**: LLaMA 2's spectrum includes various parameter sizes and model types, both pretrained and fine-tuned. Its primary function lies in text input and generation, underscoring its specialization in natural language processing.

The launch of LLaMA 2 by Meta, in partnership with Microsoft, marks a pivotal advancement in the sphere of large language models. It not only signifies a leap in AI capabilities but also paves the way for innovative research and commercial applications in the rapidly evolving AI landscape.

## Anticipating LLaMA 3: The Next Leap in Language AI?

LLaMA 3, the upcoming installment in Meta AI's LLaMA series, is poised to redefine the boundaries of large language models. This next-generation model is anticipated to bring forth enhanced capabilities and a leap in performance, promising to revolutionize research, development, and commercial applications.

Key aspects to look forward to in LLaMA 3 include:

1. **Elevated Performance**: Building on its predecessors, LLaMA3 is expected to showcase significant advancements in various AI domains, particularly in code generation, reasoning, and strategic planning.

2. **Competing with Giants**: Rumors suggest that LLaMA 3 could rival or even surpass the capabilities of GPT-4, indicating a potential major milestone in the evolution of language AI.

3. **Imminent Arrival**: The AI community is abuzz with anticipation as LLaMA 3's release is slated for early 2024, marking it as a highly awaited event in the tech world.

The arrival of LLaMA 3 represents not just another step but a giant leap forward in the journey of language AI. With its enhanced features and capabilities, it is set to unlock new horizons for a multitude of applications, further cementing Meta AI's role as a trailblazer in the field.

Apple, famously secretive, often leaves us wondering: are they really listening, or just marching to their own beat? Apple? Apple? Are you there? Hello? 🤣

## Choosing LLaMA for the MLX Journey: A Strategic Move

LLaMA has become a buzzword in AI discussions, and rightly so. It's not just chatter; LLaMA's emergence as a versatile, large language model, coupled with its open-source nature, makes it a standout choice for our exploration in the MLX Book. Its accessibility ensures anyone can leverage its capabilities without cost constraints.

The decision to integrate LLaMA into our MLX narrative is strategic. It represents a convergence of two formidable forces in AI: the cutting-edge technology of Meta's LLaMA and the innovative prowess of MLX. This synergy offers a unique opportunity to delve deep into the realm of AI, demonstrating practical applications with real-world impact.

Moreover, the open accessibility of LLaMA's weights marks a significant shift. It empowers you to run your own large language model locally, fine-tune it with your data, and tailor it to your specific needs. This flexibility, along with the benefits of privacy and security, elevates LLaMA from a mere tool to an essential component in our MLX exploration.

Indeed, the integration of LLaMA into the MLX Book is more than just a choice; it's a significant, forward-thinking move in our AI journey.

## Demystifying Open-Sourcing in AI: Weights vs. Code

Open-sourcing AI models, while a burgeoning concept, plays a pivotal role in the AI landscape. It's essential to grasp what this entails: essentially, it's the practice of making a model's weights publicly accessible. This contrasts sharply with traditional models, which were typically proprietary and limited to the company that developed them.

A common misconception, especially among non-developers, is equating open-sourcing with making the model's code public. However, open-sourcing pertains to the model's weights, not its code.

Consider the calls for OpenAI to open-source GPT-4. These appeals are for the release of the model's weights, not its underlying code. The GPT architecture, in most cases, is already established or assumed to be akin to existing frameworks, rendering the code less critical compared to the weights.

To clarify, when we talk about 'weights', we're essentially referring to the parameters of the model:

```text
parameters = weights + biases
```

Technically, weights include both the parameters and biases, but colloquially, these terms are often used interchangeably.

Think of _weights_ or _parameters_ as the amassed knowledge the model employs for making predictions. The greater the number of parameters, the more nuanced and intelligent the model's predictions are. So, if you've been keeping up with our books, remember: in the world of AI, open-sourcing models means unlocking the doors to their weights, the heart of their predictive prowess.

Just one additional insight for clarity: pretrained models come with saved weights, collectively known as a _checkpoint_, akin to a saved state in a video game. This checkpoint represents a snapshot of the model's weights at a specific moment in time. That's why pretrained models are often referred to as checkpoints, and why GPT versions are listed with dates by OpenAI, indicating their respective snapshot timings.

Consider the process of pretraining a model akin to a meticulous search for the most optimal combination of all these weights, aimed at yielding the best possible outcomes. This underlines the significance of pretraining and explains why it demands considerable time and computational resources. It's not just about accumulating data; it's a strategic exercise in fine-tuning a model to its peak potential.

## LLaMA vs. Traditional Transformer Architecture: Evolving the Paradigm

Assuming your familiarity with the attention mechanism and transformer model architecture from our previous book, we'll build upon that foundation here. We delved into these concepts extensively in Part III of the first book, so a quick review there might be beneficial if you need a refresher.

Now, let's explore the key enhancements LLaMA brings to the traditional transformer architecture. LLaMA is grounded in the transformer architecture but includes several pivotal improvements:

- **RMSNorm Normalization**: Unlike the original architecture, which normalizes the output of each transformer sub-layer, LLaMA employs `RMSNorm` to normalize the input. This change significantly enhances training stability. You will see this in action in the code.

- **SwiGLU Activation Function**: The standard ReLU non-linearity has been replaced with `SwiGLU`. This swap boosts overall performance, offering a more effective activation function. Again, you will see this in action in the code.

- **Rotary Positional Embeddings (RoPE)**: LLaMA departs from using absolute positional embeddings. Instead, it integrates rotary positional embeddings at each network layer, refining its understanding of word positions within a sequence. You got it: you'll see this in action in the code.

LLaMA shares foundational similarities with the Generative Pre-Trained Transformer (GPT) models from OpenAI and Google's Bidirectional Encoder Representations from Transformers (BERT). These large language models, including LLaMA, are all offshoots of the transformer neural network architecture, underpinned by deep learning principles.

In essence, if you're adept with the Transformer architecture's code, LLaMA's code will be accessible to you. From an object-oriented perspective, think of LLaMA as a subclass of Transformer, enriched with its own polymorphic traits. This mirrors how MLX extends and enriches functionalities over other frameworks. Object orientation isn't just a programming concept; it's a powerful lens through which to understand and learn about diverse subjects, including AI architectures like LLaMA.

## Recap: The Transformer Architecture in a Nutshell

Before diving into how LLaMA tweaks the transformer model, let's briefly revisit the core components of building a transformer model from scratch. While we covered this in our first book, understanding LLaMA's approach requires a solid grasp of these foundational elements:

1. **RMSNorm (Root Mean Square Layer Normalization)**:
   - `class RMSNorm(nn.Module)`: This module replaces the traditional Layer Normalization used in original transformers. RMSNorm normalizes the input of each sub-layer within the transformer blocks, aiming to maintain stability during training by preventing the internal covariate shift.

2. **Attention Mechanism**:
   - `class Attention(nn.Module)`: The attention mechanism is the heart of the transformer model. It calculates the weighted importance of different parts of the input data. This module particularly deals with how the model pays 'attention' to different parts of the input sequence when generating an output, ensuring that relevant context is not lost.

3. **FeedForward Network**:
   - `class FeedForward(nn.Module)`: This component consists of fully connected layers used in each transformer block. It processes the output from the attention mechanism. In transformers, this is typically a simple two-layer neural network with an activation function.

4. **Transformer Block**:
   - `class TransformerBlock(nn.Module)`: Each transformer block combines the attention and feedforward networks with residual connections and normalization. It’s the fundamental building block of the transformer architecture, where the actual processing and transformations of inputs occur.

5. **LLaMA Model Structure**:
   - `class LLaMA(nn.Module)`: This is the overarching module that encapsulates the entire LLaMA model. It stitches together multiple Transformer Blocks, embedding layers, and output layers to form the complete model. Here, the unique aspects of LLaMA, like RMSNorm, SwiGLU activation, and RoPE, integrate seamlessly with the traditional transformer architecture.

Understanding each of these components is crucial as they form the skeleton of the transformer architecture. LLaMA builds upon this foundation, introducing its enhancements to improve performance, such as the RMSNorm for stability and SwiGLU for efficient activation, while retaining the core principles of the transformer model.

By dissecting these elements, we can appreciate how LLaMA extends and refines the transformer model, making it suitable for a broader range of applications and demonstrating the potential for evolution in AI model design.

## LLaMA in MLX: A Deep Dive into the Code

Alright, let's delve into the code from Apple's MLX Examples repository.

### ModelArgs: The Foundation of a Transformer Model in MLX

```python
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True
```

The first code snippet is a foundational part of setting up a model in MLX, specifically for a transformer-like architecture such as LLaMA. 

1. **Import Statements**:
   - `import glob`, `import json`, `import time`: These are standard Python libraries for file operations (`glob`), and handling JSON data (`json`).
   - `from dataclasses import dataclass`: This imports the `dataclass` decorator from Python's dataclasses module, which is used for creating classes that primarily store data.
   - `from pathlib import Path`: `Path` is used for file system path manipulations, making file handling more intuitive.
   - `from typing import Optional, Tuple`: These are type hinting tools from Python's typing module, used for specifying the expected data types of function arguments or return values.
   - `import mlx.core as mx`, `import mlx.nn as nn`: These import statements bring in MLX's core functionality (`mlx.core`) and neural network components (`mlx.nn`), essential for building and operating the model.
   - `from mlx.utils import tree_unflatten`: This utility function from MLX is used for manipulating data structures, particularly for 'unflattening' a nested structure.
   - `from sentencepiece import SentencePieceProcessor`: _SentencePiece_ is a library for subword tokenization, which is crucial for handling text data in models like LLaMA.

2. **ModelArgs Data Class**:
   - `@dataclass`: This decorator is used to create a simple data class called `ModelArgs`. This class will hold the configuration parameters for the LLaMA model.
   - The attributes of `ModelArgs` include:
     - `dim`: Dimensionality of the model, often the size of the embeddings.
     - `n_layers`: Number of layers in the transformer model.
     - `head_dim`: Dimension of each head in the multi-head attention mechanism.
     - `hidden_dim`: Dimension of the hidden layers in the feedforward network.
     - `n_heads`: Number of heads in the multi-head attention mechanism.
     - `n_kv_heads`: Number of key/value heads in the multi-head attention, potentially for a modified attention mechanism.
     - `norm_eps`: A small epsilon value used for numerical stability in normalization layers.
     - `vocab_size`: The size of the vocabulary used by the model.
     - `rope_theta`: A parameter for rotary positional embeddings (RoPE).
     - `rope_traditional`: A boolean indicating whether to use traditional RoPE or a modified version.

This code snippet is foundational for setting up a transformer model in MLX. It shows the import of necessary libraries and the definition of a data class to hold model configuration parameters, illustrating the initial steps in building a neural network with specific characteristics and dimensions.

### RMSNorm: A Custom Normalization Layer

In this section of our exploration into LLaMA, we're going to look at a custom normalization layer, specifically the `RMSNorm` class, which is a variant of the standard normalization techniques used in transformer models. Let's dive into the details:

```python
class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output
```

1. **Class Initialization (`__init__`)**:
   - `dims`: This parameter specifies the dimension of the layer to be normalized.
   - `eps`: A small epsilon value (`1e-5`) for numerical stability, preventing division by zero.
   - `self.weight`: Initialized as a vector of ones with the same dimension as `dims`. This will be used as a scaling factor for the normalized output.

2. **Normalization Function (`_norm`)**:
   - This method performs the actual normalization process.
   - `mx.rsqrt(...)`: Computes the reciprocal of the square root. Here, it's applied to the mean squared value of the input `x`, along with the last dimension (`-1`), ensuring the operation is applied across the appropriate axis.
   - `self.eps`: Added for numerical stability.
   - The result is that each element of `x` is scaled inversely proportional to the root mean square of its corresponding vector, effectively normalizing the data.

3. **Callable Method (`__call__`)**:
   - This is where the normalization logic is applied to the input `x`.
   - `x.astype(mx.float32)`: Ensures the input is in the correct data type for the computation.
   - The normalized output is then scaled by `self.weight`, and the data type is reverted to the original input's data type.
   - This method makes `RMSNorm` a callable object, allowing us to use it like a function in the neural network.

The introduction of `RMSNorm` in LLaMA represents an improvement over the standard normalization techniques. By normalizing the input instead of the output of each sub-layer, and using a root mean square approach, `RMSNorm` helps in stabilizing the training process, which is particularly crucial for deep and complex models like LLaMA. This stability is key to achieving high performance in various natural language processing tasks.

### Attention Mechanism: The Heart of the Transformer Model

Let's dive into the `Attention` class, a crucial component of the LLaMA model, which is based on the transformer architecture. We're going to dissect this class to understand how it works.

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(
            args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)
```

**Initialization (`__init__`)**:
- **Model Parameters**: The class takes `ModelArgs` as an argument to configure various aspects of the attention mechanism.
- **Head Configurations**: `self.n_heads` and `self.n_kv_heads` are set based on the provided arguments. These define the number of attention heads and key/value heads, respectively.
- **Weight Matrices for Queries, Keys, and Values**: `self.wq`, `self.wk`, and `self.wv` are linear layers that transform the input into queries, keys, and values for the attention mechanism.
- **Output Weight Matrix**: `self.wo` is another linear layer that transforms the output of the attention computation back into the desired dimension.
- **Rotary Positional Embedding (RoPE)**: `self.rope` is an instance of the rotary positional embedding, enhancing the model's understanding of word positions.

**Callable Method (`__call__`)**:
- **Input Processing**: The input `x` is first processed through the query, key, and value weight matrices.
- **Reshaping for Attention Heads**: The transformed queries, keys, and values are reshaped and transposed to prepare them for multi-head attention computation.
- **Repeat Function**: This function repeats the keys and values to match the number of attention heads.
- **Handling Cache for Incremental Decoding**: If a cache is provided, it is used to store past keys and values, enabling incremental decoding. This is especially useful in tasks like language generation.
- **Computing Scores**: Attention scores are computed as the dot product of queries and keys, scaled by the square root of the dimension. If a mask is provided, it's added to the scores to prevent attention to certain positions.
- **Softmax and Output Generation**: The softmax function is applied to the scores, and the result is used to weight the values. The output is then combined using the output weight matrix.

This `Attention` class encapsulates the core mechanism of how attention is computed in the LLaMA model. It demonstrates an advanced implementation of the attention mechanism, including the integration of rotary positional embeddings and considerations for incremental decoding, which are crucial for handling sequential data effectively in tasks like language modeling and translation.

#### Understanding Rotary Positional Embeddings (RoPE) in LLaMA

One of the innovations in the LLaMA model, and in transformer models in general, is the use of rotary positional embeddings (RoPE). This is a key concept to understand, as it plays a critical role in how the model processes and interprets sequences of data.

In transformer models, understanding the order or position of words in a sentence is crucial. Traditional models use positional embeddings added to the input embeddings to give the model a sense of word order. However, these standard positional embeddings can have limitations, especially in capturing relative positions or in large sequences.

Rotary Positional Embeddings offer a solution to these issues. The core idea behind RoPE is to encode the position information into the embeddings in a way that preserves the relative positioning of the words or tokens.

RoPE works by rotating the attention scores in a way that depends on the position. The key innovation is how it encodes the relative position information:

1. **Rotational Encoding**: Instead of simply adding position information to each token's embedding, RoPE applies a rotation to the embedding space. This rotation is dependent on the position of the tokens, effectively encoding their relative positions.

2. **Relative Positioning**: RoPE allows the model to understand the relative positions of tokens more effectively. This means that the model doesn't just know which token comes before another; it understands how far apart they are in the sequence.

3. **Integration with Attention Mechanism**: RoPE is integrated into the attention mechanism of the transformer. During the attention calculation, the query and key vectors are rotated based on their positions before their dot product is computed. This rotation alters the attention scores in a way that reflects the relative positions of the tokens.

##### Advantages of Rotary Positional Embeddings

- **Improved Relative Position Understanding**: RoPE provides a more nuanced understanding of relative positions within a sequence, which is particularly useful in tasks like language modeling and machine translation.
  
- **Efficiency in Longer Sequences**: Traditional positional embeddings can struggle with longer sequences due to their fixed nature. RoPE's dynamic approach makes it more effective for handling longer sequences of data.

- **Compatibility with Incremental Decoding**: RoPE is particularly compatible with tasks that involve incremental decoding, like language generation, as it encodes position information in a way that is conducive to adding one token at a time.

In summary, Rotary Positional Embeddings represent a significant advancement in the way transformer models handle position information. By encoding relative positions more effectively and efficiently, RoPE enhances the model's understanding of sequences, leading to better performance in a range of natural language processing tasks.

### FeedForward Network: The Core of the Transformer Block

The FeedForward class is a crucial component in the LLaMA model, encapsulating the feedforward network within each transformer block. Let's explore its structure and functionality in detail.

```python
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))
```

#### Initialization (`__init__`):

- **Linear Layers**: The class initializes three linear layers: `self.w1`, `self.w2`, and `self.w3`. 
  - `self.w1` and `self.w3` act as the first and third layers of the feedforward network, transforming the input dimension (`args.dim`) to the hidden dimension (`args.hidden_dim`) and vice versa.
  - `self.w2` serves as the middle layer, transforming data back from the hidden dimension to the input dimension.
- **No Biases**: Note that biases are not used in these layers (`bias=False`). This is a design choice that can vary based on model architecture.

#### Callable Method (`__call__`):

- **Feedforward Computation**: The `__call__` method defines how the input `x` is processed through the feedforward network.
- **Activation Function**: `nn.silu` (Sigmoid Linear Unit) is used as the activation function. This function, applied to the output of `self.w1(x)`, introduces non-linearity into the network, which is essential for the model to capture complex patterns in the data.
- **Element-Wise Multiplication**: The output of `nn.silu(self.w1(x))` is element-wise multiplied with `self.w3(x)`. This is an interesting architectural choice, as it combines two transformed versions of the input.
- **Output Generation**: Finally, this combined output is passed through `self.w2`, resulting in the final output of the feedforward network.

##### SiLu Activation Function: A Closer Look

The Sigmoid Linear Unit (SiLU), also known as the Swish function, is an activation function used in neural networks. It is defined mathematically as:

![silu1.png](images%2Fsilu1.png)

where `σ(x)` is the sigmoid function:

![silu2.png](images%2Fsilu2.png)

The key characteristic of SiLU is that it combines aspects of linear and sigmoid functions. When `(x)` is large, SiLU behaves like a linear function, allowing the network to capture relationships in high-activation regions efficiently. When`(x)` is small, it approximates a sigmoid, controlling the activation in low-activation regions.

This dual nature makes SiLU effective for deep learning models, including in tasks like language processing and image recognition. It can help the network learn complex patterns while maintaining efficient computation. SiLU has been shown to sometimes outperform other common activation functions like ReLU (Rectified Linear Unit) in certain neural network architectures.

#### Significance in LLaMA:

- The FeedForward class in LLaMA adds a layer of complexity to the traditional feedforward design in transformers. 
- By incorporating an additional linear transformation (`self.w3`) and an element-wise multiplication step, LLaMA’s feedforward network is more sophisticated than the standard two-linear-layer design. 
- This complexity allows the model to learn more intricate relationships in the data, which can be particularly beneficial in handling the nuances of natural language.

In essence, the FeedForward class in LLaMA showcases an evolved version of the standard transformer feedforward network. Its design reflects a balance between complexity and computational efficiency, contributing to LLaMA's ability to effectively process and learn from large volumes of language data.

### Transformer Block: The Core Building Block of the LLaMA Model

In this section, we're examining the `TransformerBlock` class, a fundamental building block of the transformer model, particularly in the context of LLaMA. This class encapsulates both the attention mechanism and the feedforward network, along with normalization layers. 


```python
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache

```

**Initialization (`__init__`)**:
- **Component Initialization**: The block initializes key components like the attention mechanism (`self.attention`), the feedforward network (`self.feed_forward`), and two RMSNorm layers (`self.attention_norm` and `self.ffn_norm`).
- **Model Parameters**: It takes `ModelArgs` as input, which provides necessary parameters like the number of heads (`n_heads`) and the dimensionality (`dim`) for the attention and feedforward layers.

**Callable Method (`__call__`)**:
- **Attention Computation**: The input `x` is first normalized using `self.attention_norm` and then passed to the attention layer. The attention layer can optionally take a `mask` and a `cache` for masked attention and incremental decoding, respectively.
- **Residual Connection and Feedforward Network**: The output from the attention layer (`r`) is added to the original input (`x`) in a residual connection. The result (`h`) is then normalized and passed through the feedforward network.
- **Final Output**: Another residual connection adds the output of the feedforward network back to `h`, producing the final output of the transformer block.

#### Significance in the Transformer Architecture:

- **Normalization Layers**: The use of RMSNorm instead of traditional LayerNorm is a distinctive aspect of LLaMA's design, aimed at stabilizing the training process.
- **Residual Connections**: These are crucial for preventing the vanishing gradient problem in deep networks, allowing the model to learn effectively even with many layers.
- **Flexibility with Masking and Caching**: The ability to include masks and cache previous computations makes this block versatile for different tasks, like language generation where past outputs influence future predictions.

The `TransformerBlock` class in LLaMA is a comprehensive unit that integrates key transformer components. Its design reflects an optimized balance between complexity and functionality, contributing to the overall efficacy of the LLaMA model in processing and understanding complex language data.

### LLaMA Model: The Complete Transformer Model

The `Llama` class is the final piece of the puzzle, encapsulating the entire LLaMA model. Let's explore its structure and functionality in detail.

```python
class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        x = self.tok_embeddings(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.output(x)

    def generate(self, x, temp=1.0):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.tok_embeddings(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            # We store the per layer cache in a simple python list
            cache.append(c)
        x = self.norm(x)
        # We only care about the last logits that generate the next token
        y = self.output(x[:, -1])
        y = sample(y)

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.tok_embeddings(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = sample(self.output(x[:, -1]))

            yield y
```

This class integrates all the components we've discussed - embeddings, transformer blocks, normalization, and output layers.

**Initialization (`__init__`)**:
- **Embedding Layer**: `self.tok_embeddings` is an embedding layer that converts input tokens into continuous vectors. It uses the vocabulary size and embedding dimension from `args`.
- **Transformer Blocks**: `self.layers` consists of a series of `TransformerBlock` instances, as defined by `args.n_layers`. These are the core processing units of the model.
- **Final Normalization**: `self.norm` is an RMSNorm layer applied to the output of the last transformer block.
- **Output Layer**: `self.output` is a linear layer that maps the final transformer output back to the vocabulary space.

**Forward Method (`__call__`)**:
- **Mask Creation**: A causal mask is created for the attention mechanism to ensure that predictions for a token can only depend on previous tokens.
- **Embedding and Processing Through Layers**: Input tokens are embedded and sequentially processed through each transformer block.
- **Final Normalization and Output**: The output from the last layer is normalized and passed through the output layer to produce the final logits.

**Text Generation Method (`generate`)**:
- **Sampling Function**: `sample` is a function defined within `generate` for drawing samples from the output distribution. Temperature `temp` is used to control the randomness of the sampling.
- **Cache Mechanism for Incremental Decoding**: Caches are used to store intermediate states for efficient text generation.
- **Yield Generated Tokens**: The model generates one token at a time in a loop, reusing the cached states for efficiency. The `yield` statement is used to return each generated token.

### Significance in LLaMA:

- The `Llama` class encapsulates the entire LLaMA model, from input to output, including the text generation process.
- The generation method demonstrates an important aspect of transformer models in NLP tasks: the ability to incrementally generate text token by token.
- The use of RMSNorm and the specific design choices in the transformer blocks reflect LLaMA's advancements over traditional transformer models.

In essence, the `Llama` class is where all the components of the LLaMA model come together, demonstrating how the model processes input data and generates output, be it for standard forward passes or for generating text in an autoregressive manner.

### Helper Functions - `generate`

The `generate` function is a helper function that takes a trained model and a prompt as input and generates text based on the model's predictions. 

```python
def generate(prompt, temp=0.7, max_tokens=200, write_every=100):
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(prompt)])
    skip = 0
    tokens = []
    response = ''

    print("[INFO] Generating response...")
    print("[INFO] Prompt: {}".format(prompt))

    for token in model.generate(x, temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)

        if len(tokens) >= max_tokens:
            break
        elif (len(tokens) % write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            response += s[skip:]
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    response += s[skip:]
    print(s[skip:], flush=True)
    return response
```

1. **Initialization**:
   - The function starts by converting the given `prompt` into a tokenized format suitable for the model.
   - Initial variables like `skip`, `tokens`, and `response` are set up for tracking the generation process.

2. **Informative Print Statements**:
   - The function now includes print statements to inform the user that the response generation is starting and to display the provided prompt.

3. **Generation Loop**:
   - The loop iterates over the tokens generated by the model.
   - Each token is appended to the `tokens` list.

4. **First Token Evaluation**:
   - After the first token is generated, `mx.eval(token)` forces its evaluation to get the prompt processing time.

5. **Conditional Checks and Response Update**:
   - The loop breaks if the number of tokens reaches `max_tokens`.
   - Every `write_every` number of tokens, the generated text is evaluated and printed. This provides real-time feedback on the text being generated.
   - The response string is continuously updated with the new text.

6. **Final Output**:
   - After the generation loop, the entire set of tokens is evaluated.
   - The final generated text is decoded and appended to the `response` string, which is then printed.
   - The function returns the complete response.

The use of `mx.eval` ensures that the model's lazy computations are executed efficiently, while the print statements keep the user informed of the ongoing generation process. This approach is particularly useful for generating longer pieces of text where immediate feedback can enhance the user experience.

### Helper Functions - `sanitize_config`

Let me explain the purpose and functionality of the `sanitize_config` function in the context of setting up a model like LLaMa. This function is designed to clean and update the model configuration dictionary based on the provided weights and default values.

```python
def sanitize_config(config, weights):
    config.pop("model_type", None)
    n_heads = config["n_heads"]
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    if "hidden_dim" not in config:
        config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    if config.get("vocab_size", -1) < 0:
        config["vocab_size"] = weights["output.weight"].shape[-1]
    if "rope_theta" not in config:
        config["rope_theta"] = 10000
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)
    return config

```

**Functionality of `sanitize_config`**:
- **Remove Unnecessary Entries**: The function starts by removing entries like `"model_type"` from the configuration dictionary. These might be irrelevant for our specific model setup.
- **Set Default Number of Key/Value Heads**: If `"n_kv_heads"` is not specified, it defaults to the number of heads (`n_heads`).
- **Calculate Head Dimension**: If `"head_dim"` is missing, it's computed as the dimensionality divided by the number of heads. This ensures each head in the multi-head attention mechanism has the correct dimension.
- **Set Hidden Dimension**: For the hidden dimension of the feedforward network, if it's not in the config, the function infers it from the shape of the feedforward layer weights.
- **Determine Vocabulary Size**: If the vocabulary size is not specified or set to a negative value, it's inferred from the shape of the output layer weights. This ensures compatibility with the model's vocabulary.
- **Set Default Rotary Positional Embeddings Theta**: If `"rope_theta"` is not provided, a default value of 10000 is assigned. This parameter is crucial for the rotary positional embeddings.
- **Remove Unused Config Entries**: The function also cleans up any other unused configuration entries, such as `"multiple_of"` or `"ffn_dim_multiplier"`, to avoid potential conflicts or confusion during model initialization.

**Purpose of Sanitization**:
- **Adaptability**: This function makes the configuration adaptable to the provided weights, ensuring that the model initializes correctly even if some details are not explicitly specified in the configuration.
- **Robustness**: It adds robustness to the model setup process, handling cases where the configuration might be incomplete or slightly mismatched with the weights.
- **Simplicity**: By providing default values and removing unnecessary entries, `sanitize_config` simplifies the model configuration process, making it more user-friendly and less error-prone.

In summary, the `sanitize_config` function is a crucial step in preparing the configuration for building the LLaMa model. It intelligently adjusts the configuration to align with the provided weights and fills in any gaps with sensible defaults. This function exemplifies the kind of preprocessing often needed in complex model setups to ensure smooth and error-free initialization.

### Helper Functions - `load_model`

Let me walk you through the `load_model` function. This function is critical for loading the LLaMA model and its associated tokenizer from a given path. It handles different scenarios, like unsharded or sharded weights, and sets up the model configuration.

```python
def load_model(model_path):
    model_path = Path(model_path)

    unsharded_weights_path = Path(model_path / "weights.npz")
    if unsharded_weights_path.is_file():
        print("[INFO] Loading model from {}.".format(unsharded_weights_path))
        weights = mx.load(str(unsharded_weights_path))
    else:
        sharded_weights_glob = str(model_path / "weights.*.npz")
        weight_files = glob.glob(sharded_weights_glob)
        print("[INFO] Loading model from {}.".format(sharded_weights_glob))

        if len(weight_files) == 0:
            raise FileNotFoundError("No weights found in {}".format(model_path))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf).items())

    with open(model_path / "config.json", "r") as f:
        config = sanitize_config(json.loads(f.read()), weights)
        quantization = config.pop("quantization", None)
    model = Llama(ModelArgs(**config))
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(tree_unflatten(list(weights.items())))
    tokenizer = SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
    return model, tokenizer
```

**Model Path Handling**:
- The path to the model is converted into a `Path` object, facilitating file path operations.

**Loading Unsharded Weights**:
- The function first checks for the presence of unsharded weights (a single file containing all weights).
- If found, it prints an informative message and loads the weights using `mx.load`.

**Handling Sharded Weights**:
- If unsharded weights are not found, it searches for sharded weights (weights split across multiple files).
- It loads each of these weight files, updating a dictionary that collectively holds all the weights of the model.

**Loading Model Configuration**:
- The configuration file (`config.json`) is read, and the contents are passed to `sanitize_config` function along with the weights. This adjusts the configuration based on the actual weights and adds any missing information.
- If there's any quantization configuration in the model's config, it is applied to the model using `nn.QuantizedLinear.quantize_module`.

**Model Instantiation**:
- A `Llama` model instance is created using the sanitized configuration.
- The weights are updated in the model using `tree_unflatten`, which correctly aligns them with the model's architecture.

**Tokenizer Loading**:
- The SentencePiece tokenizer is loaded from the specified model path. This tokenizer is crucial for encoding and decoding the inputs and outputs of the model.

**Return Model and Tokenizer**:
- Finally, the function returns both the loaded model and the tokenizer.

This function is a comprehensive solution for initializing the LLaMA model and its tokenizer from saved files. By handling both unsharded and sharded weights, it ensures flexibility and adaptability in different deployment scenarios. The additional steps of sanitizing the configuration and handling quantization demonstrate careful attention to the nuances of model loading, ensuring the model is ready for use with the correct configuration and weights. This function is essential for anyone who wants to work with pre-trained LLaMA models, providing a straightforward and reliable way to load the model and tokenizer for various NLP tasks.

### The WebUI: A User-Friendly Interface for LLaMA Using Streamlit

Now, let's explore the Streamlit application that provides a user-friendly interface for interacting with the LLaMA model. This application is designed to showcase the capabilities of LLaMA in a simple and intuitive way.

#### Caching the Model in Streamlit with `@st.cache_resource`

In the context of developing a Streamlit application for our LLaMa-based chatbot, efficient management of resources is crucial. Loading a large model like LLaMA can be resource-intensive, and doing it repeatedly for every interaction in the app is inefficient. To address this, we use Streamlit's `@st.cache_resource` decorator.

```python
@st.cache_resource
def load_cached_model(model_path):
    return load_model(model_path)
```

**Function Explanation**:
- **Purpose**: The function `load_cached_model` is designed to load the LLaMA model from a given path. It uses the `load_model` function, which initializes the model and tokenizer.
- **Caching with `@st.cache_resource`**: The decorator `@st.cache_resource` is applied to `load_cached_model`. This tells Streamlit to cache the loaded model and tokenizer in memory.
- **Efficiency**: Once the model is loaded for the first time, Streamlit stores it in the cache. Subsequent calls to `load_cached_model` with the same `model_path` will quickly retrieve the model from the cache instead of reloading it from disk.

**Advantages of Using `@st.cache_resource`**:
- **Performance**: By caching the model, we significantly reduce the load time after the first execution. This improves the overall performance and user experience of the Streamlit app.
- **Resource Management**: It helps in managing system resources more effectively by avoiding redundant model loading.
- **Simplicity**: The decorator abstracts away the complexity of implementing a caching mechanism, making the code cleaner and easier to maintain.

In summary, the use of `@st.cache_resource` in our Streamlit application is a strategic choice to enhance efficiency and responsiveness. It ensures that the heavy lifting of loading the LLaMa model is done just once, allowing for smoother interactions in the chatbot application.

An important reminder about working with GPTs, especially in the context of Streamlit: Streamlit is rapidly evolving, and GPTs may not always be up-to-date with the latest changes. They might recommend solutions like `st.cache(allow_output_mutation=True)` or `@st.experimental_singleton`, which could be deprecated.

If you encounter deprecation warnings in your code, I strongly advise checking the [Streamlit documentation](https://docs.streamlit.io/en/stable/) for the most current updates and practices. Streamlit typically includes helpful links in its warning messages, so it's wise not to overlook them. Staying informed about the latest developments ensures that your Streamlit applications remain efficient, robust, and in line with the best current practices.

#### Interactive Chatbot with `st.chat_message`

We're enhancing the Streamlit chatbot application to make it more interactive and visually appealing by adding avatars for both the user and the chatbot. 

##### The `generate_response` Function

```python
def generate_response(user_input):
    if user_input:  # Check if the input is not empty
        full_prompt = SYSTEM_MESSAGE + f"\n\n[INST] {user_input} [/INST]\n"
        return generate(full_prompt, temp=temp, max_tokens=max_tokens, write_every=WRITE_EVERY)
    return ""
```

- **Purpose**: This function handles the generation of responses based on the user's input.
- **Input Check**: It first checks if the `user_input` is not empty. If it's empty, the function returns an empty string.
- **Response Generation**: If there is input, the function constructs a full prompt by appending the `SYSTEM_MESSAGE` and the user's input in the designated format. It then calls the `generate` function to produce a response.

```python
if __name__ == "__main__":
    SEED = 42
    MODEL_PATH = "/Users/wankyuchoi/cwk-llm-models/llama-2-7b-chat-mlx"
    WRITE_EVERY = 100
    SYSTEM_MESSAGE = "<<SYS>>Your name is Menny, a cynical teenager AI assistant.<</SYS>>"
    MODEL_NAME = "Menny Llama"
    MODEL_AVATAR = "./images/menny-avatar.png"
    HUMAN_AVATAR = "./images/human-avatar.png"

    mx.random.seed(SEED)
    model, tokenizer = load_cached_model(MODEL_PATH)

    # Streamlit UI setup
    st.sidebar.title("Chatbot Settings")
    max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 200)
    temp = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)

    st.title(MODEL_NAME)
    st.image(MODEL_AVATAR, width=500)

    user_input = st.chat_input("Your Message")
    response = generate_response(user_input)

    if response:  # Display the response only if it's not empty
        human_message = st.chat_message('human', avatar=HUMAN_AVATAR)
        human_message.write(user_input)
        ai_message = st.chat_message('assistant', avatar=MODEL_AVATAR)
        ai_message.write(response)
```

- **Model and Tokenizer Loading**: The model and tokenizer are loaded using the `load_cached_model` function. This is done once when the script starts.
- **Streamlit UI Setup**: The sidebar is set up with sliders to control `max_tokens` and `temp`, which influence the length and variability of the chatbot's responses.
- **Title and Avatars**: The main page displays the title (MODEL_NAME) and the chatbot's avatar image.
- **Chat Input**: `st.chat_input` creates an input field for the user to type their message.
- **Response Handling**: The `generate_response` function is called with the user's input to get the chatbot's response.
- **Displaying Chat Messages**: 
  - If there's a response, the user's message and the chatbot's response are displayed as chat messages, complete with respective avatars. 
  - `st.chat_message` creates a chat message bubble. For the user, `human` is passed as an identifier, and for the chatbot, `assistant` is used. 
  - The `avatar` parameter sets the respective avatars for the user and chatbot messages.

This code snippet turns the Streamlit app into a more engaging chatbot interface by visually representing the conversation with avatars, akin to modern messaging apps. The use of `st.chat_message` adds a layer of realism and improves the overall user experience.

## Enhancing Menny Llama: Unlocking New Features
![menny-llama.png](images%2Fmenny-llama.png)
In developing Menny Llama, I deliberately left room for expansion, knowing that with tools like Streamlit, there's a world of possibilities to enhance our chatbot. Here are some ideas for features that you, as a budding developer, can easily implement to elevate Menny's capabilities:

1. **Context Retention**:
   - Utilize Streamlit's session state to remember the flow of the conversation. Store each exchange in a dictionary and maintain them in a list. This will allow Menny to recall previous interactions, providing contextually relevant responses. Remember to include the system message in this context to maintain Menny's unique persona.

2. **Chat History Display**:
   - Implement a feature to show the entire chat history in a specific section of the app. Streamlit has built-in capabilities for this, making it a straightforward addition. This enhances user experience by allowing them to review past interactions.

3. **Editable System Message**:
   - Introduce a feature to display and modify the system message through a text box. This gives users the flexibility to change Menny's system message as they see fit, adding a layer of customization to the chatbot.

4. **Multiple Model Loaders**:
   - Expand Menny's versatility by adding a feature to load different models. Incorporate a dropdown menu in the sidebar for model selection and set up the app to load the chosen model. Consider implementing an auto-loader that can fetch and load models directly from HuggingFace's model hub.

5. **Integrating Text-to-Speech and Speech-to-Text**:
   - Leverage APIs like `whisper` and ElevenLabs to add voice chat functionalities. Implementing these features is simpler than it might initially appear.  

6. **LangChain Integration**:
   - LangChain is a powerful, framework-agnostic library for building LLM applications. By integrating LangChain, you can create a more flexible chatbot application that easily switches between different models and tokenizers. Check out my PippaGPT-MLX project on GitHub as a reference: https://github.com/neobundy/pippaGPT-MLX

7. **LoRA Training**:
   - Why not revisit the LoRA training method we explored in the first book? Applying LoRA training to Menny could significantly enhance its learning capabilities and performance.

These ideas are just the beginning. The beauty of coding is that your creativity sets the boundaries. While some of these features may seem complex, they're quite feasible to implement, often in less time than you might expect. So go ahead, take Menny to the next level, and watch as your coding skills bring new dimensions to this chatbot project.

## Reflecting on Our Coding Odyssey: From PyTorch to MLX

![scale-one-fear-none.jpeg](..%2F..%2Fimages%2Fscale-one-fear-none.jpeg)

As we conclude the journey through both of my books, I want to emphasize a philosophy that has guided my approach: the value of self-driven learning. I believe in encouraging you to explore and create your own coding paths, as true understanding and skill in coding come from hands-on experience and personal exploration.

In my view, the essence of being a real coder is somewhat innate. While it's certainly possible to teach someone to become proficient in coding, reaching the pinnacle of greatness in coding often requires an intrinsic passion and aptitude. This might resonate with you if you feel a natural inclination towards coding, finding languages like Python not just tools but extensions of your thought process.

If you're one of those who are 'born to code', the extensive explorations we've embarked on in these books might seem superfluous. But they're designed to be comprehensive, leaving no stone unturned.

The second book, focusing on MLX, might have felt shorter compared to the first. This brevity is intentional and reflects an object-oriented approach to learning. With the foundational concepts covered in the first book, the second book serves as an extension, introducing you to the unique aspects of MLX.

As we reach the end of this path that I envisioned when I began writing these books, I encourage you to embark on your own journey in coding. I hope you found the ride enlightening and enjoyable, as I certainly did. I've learned immensely, not just from the process of writing but also from the invaluable insights provided by my GPT companions.

While this may be the conclusion of the current series, I remain open to the possibility of adding more content in the future if needed. For now, though, I bid you farewell and good luck on your AI adventures. Thank you for joining me on this journey, and I hope the knowledge you've gained serves you well in your future endeavors.

My true reward has always been the journey, not the destination. I hope you find the same fulfillment. Always remember, we started this incredible journey from just a single tensor.