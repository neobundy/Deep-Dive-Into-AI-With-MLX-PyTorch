# Deep Dive into LLaVA - Large Language and Vision Assistant Part I

![llava-gradio.png](images%2Fllava-gradio.png)

✍️ [Part I](README.md) | ✍️ [Part II](README2.md) | ✍️ [Part III ](README3.md)

**🏠 Official Website**: https://llava-vl.github.io

_**LLaVA**_, the Large Language-and-Vision Assistant, heralds a new era in multimodal AI by seamlessly integrating advanced vision encoding with the linguistic prowess of _Vicuna_. This end-to-end trained model not only pioneers general-purpose visual and language understanding but also showcases remarkable chat capabilities that echo the versatility of multimodal GPT-4, establishing a new benchmark in Science QA accuracy.

![conceptual-diagram-of-llava.png](images%2Fconceptual-diagram-of-llava.png)

**Innovations in Instruction Tuning**

Traditionally, the enhancement of zero-shot capabilities in large language models (LLMs) has been achieved through instruction tuning with machine-generated data. However, this approach has been predominantly language-centric, with multimodal domains remaining largely untapped. LLaVA breaks new ground by leveraging language-only GPT-4 generated data to train on language-image instruction-following tasks, marking a significant stride in multimodal learning.

**LLaVA Model: A Synergistic Fusion**

At its core, LLaVA combines a pre-trained _CLIP ViT-L/14_ visual encoder with the large language model _Vicuna_ through a streamlined projection matrix. This fusion enables LLaVA to process and understand both visual and textual inputs with unparalleled accuracy. The model undergoes a two-stage instruction-tuning process to enhance its feature alignment and task-specific performance, catering to a broad spectrum of applications from visual chat to Science QA.

**Groundbreaking Performance and Open-Source Commitment**

Preliminary experiments with LLaVA have been promising, demonstrating its capability to mimic multimodal GPT-4 behaviors and achieving a relative score of 85.1% against GPT-4 on a synthetic multimodal instruction-following dataset. When further fine-tuned for Science QA, LLaVA reaches a new peak accuracy of 92.53%. In line with our commitment to the AI community, we are making the visual instruction tuning data, model, and codebase publicly accessible.

**Tuning for Excellence**

LLaVA's training process is meticulously designed to optimize its multimodal capabilities. Initially, the model focuses on feature alignment through updates to the projection matrix using a subset of CC3M. Subsequently, comprehensive end-to-end fine-tuning is conducted for two distinct use cases: enhancing daily user-oriented applications through visual chat and advancing multimodal reasoning in the science domain with Science QA.

By forging a path for multimodal instruction tuning and showcasing exemplary performance, LLaVA sets the stage for future advancements in AI, where understanding transcends the boundaries of language and vision.

## LLaVA Visual Instruct 150K Dataset

![Visual Overview of LLaVA Dataset](images%2Fdataset.png)

** 👉 Dataset on Hugging Face**: [LLaVA-Instruct-150K on Hugging Face](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)

The LLaVA Visual Instruct 150K dataset represents a groundbreaking collection of GPT-generated, multimodal instruction-following data, meticulously curated to advance visual instruction tuning and foster the development of large-scale multimodal models with capabilities paralleling the vision/language integration seen in GPT-4.

**Dataset Compilation**: This invaluable dataset was assembled in April 2023, leveraging the GPT-4-0314 API to generate a rich tapestry of multimodal instructions.

![Instruction Dataset Visualization](images%2Fintruction-dataset.png)

**Insightful Visualizations**: Delve into the essence of the LLaVA Visual Instruct 150K dataset with the intuitive pie charts, each offering a vibrant breakdown of the prevalent noun-verb pairs that anchor the dataset's instructions and responses. These visual guides not only illustrate the dataset's thematic diversity but also provide quantifiable insights into the frequency of specific linguistic pairings across various instruction contexts — from everyday conversation to intricate complex reasoning. Engage with these dynamic visualizations to grasp the nuanced interplay between the verbs and nouns that drive the LLaVA model's understanding and generation of multimodal content. 

## The COCO Dataset Overview

🏠 Official Website: https://cocodataset.org

The COCO (Common Objects in Context) dataset stands as a pivotal resource for various computer vision tasks, including object detection, segmentation, and captioning. Encompassing over 330,000 images, the dataset is enriched with detailed annotations across 80 object categories, complemented by five descriptive captions for each image. This extensive collection has become a cornerstone in computer vision research, serving as a critical benchmark for evaluating machine learning algorithms. Beyond mere imagery, COCO provides comprehensive annotations that include bounding boxes, object areas, and category labels, making it an invaluable asset for training and assessing computer vision models. For those looking to leverage the COCO dataset in their projects, it is readily available for download from its official website, offering a rich dataset to advance machine learning endeavors in the realm of computer vision.

## CLIP ViT-L/14

We've already covered CLIP in the following deep dive: 

👉 [Deep Dive into CLIP](..%2FCLIP%2FREADME.md)

## LLaMA - Ancestor LLM for LLaVA

LlaMA is extensively covered in the following chapter of the 2nd book:

[Chapter 7 - Menny Llama & the End of Our Journey](..%2F..%2Fmlx-book%2F007-menny-llama-and-the-end-of-our-journey%2FREADME.md)

👉 [Deep Dive into CLIP](..%2FCLIP%2FREADME.md)

## Vicuna - Base LLM for LLaVA

**🏠 Official Website**: https://lmsys.org/blog/2023-03-30-vicuna/

At the heart of LLaVA lies Vicuna, a large language model grounded in the innovative LlaMA 2 architecture developed by Meta. Vicuna emerges as a pivotal foundation, propelling LLaVA towards achieving groundbreaking multimodal understanding by enriching its linguistic capabilities.

👉 _It's important to highlight that the latest iterations of LLaVA now include additional models, as elaborated in subsequent sections._

While the evolution of LLMs has significantly advanced chatbot technologies, leading to marvels like OpenAI's ChatGPT, the opacity in training methodologies and architecture specifics often limits further open-source development and research. Vicuna, drawing inspiration from Meta's _LLaMA_ and Stanford's _Alpaca_ project, presents itself as an open-source beacon in the realm of chatbots. The _Vicuna-13B_ model, enhanced with a meticulously curated dataset and a scalable, user-friendly infrastructure, stands out for its open-source ethos and competitive edge over counterparts like Stanford Alpaca.

**Vicuna's Development: A Community-Centric Approach**

![vicuna-workflow.png](images%2Fvicuna-workflow.png)

Vicuna's inception involved collecting approximately 70,000 conversations from _ShareGPT.com_, a platform enabling users to share their ChatGPT dialogues. This collaborative effort was further bolstered by refining Alpaca's training scripts to adeptly handle multi-turn conversations and extended sequences. Employing PyTorch FSDP and A100 GPUs, Vicuna underwent an efficient training process, culminating in a lightweight, distributed system for demo serving. A rigorous evaluation against a diverse set of questions, assessed by GPT-4, underscored Vicuna's robust response quality.

**Technological Enhancements and Open-Source Pledge**

Vicuna's training regimen incorporated several advancements:
- **Multi-turn Conversation Handling**: Adjusting training loss to favor multi-turn dialogue comprehension.
- **Memory Optimizations**: Expanding the max context length to 2048, employing techniques like gradient checkpointing and flash attention to manage the increased GPU memory demand.
- **Cost-Efficient Training**: Utilizing SkyPilot managed spot instances for a cost-effective training process, significantly reducing expenses for both the 7B and 13B models.

**Acknowledging Limitations and Moving Forward**

Despite its prowess, Vicuna shares common limitations with other LLMs, such as challenges in reasoning or mathematics and ensuring factual accuracy. To mitigate potential safety concerns, the OpenAI moderation API is used to screen inputs in the online demo. Vicuna is positioned as an open platform for future explorations aimed at overcoming these obstacles.

**Engage with Vicuna**

The online demo, intended for research and non-commercial use, adheres to the LLaMA model license, OpenAI's data use terms, and ShareGPT's privacy practices. Vicuna's source code is released under the Apache License 2.0, inviting community engagement and contribution.

## Conceptual Diagram of LLaVA

![conceptual-diagram-of-llava.png](images%2Fconceptual-diagram-of-llava.png)

The conceptual diagram illustrates the architecture of the LLaVA model, which integrates both visual and language data for multimodal understanding.

At the base, we have the **Vision Encoder**, which processes an input image `X_v`. The vision encoder transforms the raw image data into a set of feature representations `H_v`, capturing the visual information in a form that the model can work with.

Above the vision encoder is the **Projection `W`**, which serves as a bridge between the vision encoder and the language model. The projection matrix converts the feature representations `H_v` from the vision encoder into a compatible format `Z_v` for the language model. This step is crucial as it aligns the visual data with the language data, ensuring that both modalities can be processed together effectively.

On the right side, we have the **Language Instruction** input `X_q`, which is the textual component that the model needs to understand and respond to in conjunction with the visual input. This input is processed by the language model to produce its own set of feature representations `H_q`.

The central element of the diagram is the **Language Model `f_ϕ`**, a large language model like Vicuna, as mentioned in the context provided. This model takes in both the projected vision features `Z_v` and the language features `H_q` and integrates them to generate a **Language Response** `X_hat_a`.

The output is a coherent response that incorporates elements from both the visual and textual inputs. This capability allows LLaVA to perform tasks such as answering questions about an image, describing scenes, and even engaging in complex reasoning that requires understanding content from both visual and textual data.

In summary, the diagram represents how LLaVA handles multimodal inputs (images and text) to produce a language-based output, demonstrating its ability to understand and respond to multimodal data.

### Deep Dive into Projection Matrices

In the realm of multimodal models like LLaVA, projection matrices play a pivotal role in harmonizing the information from different modalities – namely visual and linguistic data. To fully grasp the nuances of projection matrices within such a context, let's revisit the concept and explore its application in multimodal learning.

**Understanding Projection Matrices**

A projection matrix, in general machine learning terms, is a linear transformation tool that maps data from one space to another. Specifically, it is used to transform the features extracted from one type of input (like an image) so that they can be effectively used in conjunction with another type of input (like text) within the same model.

In the context of the LoRA (Low-Rank Adaptation) method, which we discussed in our sidebar [LoRA-Made-Easy.md](..%2F..%2Fbook%2Fsidebars%2Flora-made-easy%2FLoRA-Made-Easy.md), projection matrices are a form of parameter-efficient learning where small, learnable matrices are inserted into a pre-trained model. These matrices adapt the pre-existing weights of the model in a low-rank format, allowing the model to learn new tasks or adapt to new data without the need for extensive retraining of all parameters.

In Transformer architectures, projection matrices are integral to the _self-attention mechanism_, facilitating the transformation of input data into distinct components—queries, keys, and values—that enable the model to selectively concentrate on pertinent segments of the input sequence. Furthermore, these matrices are responsible for producing the final attention-augmented output.

This critical functionality is reflected in the nomenclature found within code implementations. For instance, in the PyTorch implementation of a Transformer, projection matrices are designated with names that clearly indicate their roles:

```python
self.self_attn.in_proj_weight,  # Weights for input projection
self.self_attn.in_proj_bias,    # Biases for input projection
self.self_attn.out_proj.weight, # Weights for output projection
self.self_attn.out_proj.bias,   # Biases for output projection
```

Similarly, in the `clip.py` file from the Apple MLX Stable Diffusion example, the CLIP model's encoder layer uses projection matrices within its attention mechanism. The CLIP model adds biases to these projections, aligning with the original CLIP implementation:

```python
class CLIPEncoderLayer(nn.Module):
    # Transformer encoder layer from CLIP

    def __init__(self, model_dims: int, num_heads: int):
        super().__init__()
        ...
        self.attention = nn.MultiHeadAttention(model_dims, num_heads)
        # Initialize biases for attention projections to match CLIP
        self.attention.query_proj.bias = mx.zeros(model_dims)
        self.attention.key_proj.bias = mx.zeros(model_dims)
        self.attention.value_proj.bias = mx.zeros(model_dims)
        self.attention.out_proj.bias = mx.zeros(model_dims)
        ...
```

Projection matrices, whether for encoding input data into self-attention's critical components or for integrating disparate modalities in models like LLaVA, are essential for enhancing the model's interpretive and responsive capabilities, offering a testament to their versatility in various AI model architectures.

[Attention-Is-All-You-Need-For-Now](..%2F..%2Fbook%2Fsidebars%2Fattention-is-all-you-need-for-now%2FAttention-Is-All-You-Need-For-Now.md)

**Projection Matrices in LLaVA**

In a multimodal setting, the projection matrix does something analogous. LLaVA uses a vision encoder to process visual inputs (images) and a language model to process textual inputs (language instructions). Each of these components understands data in its unique way:

- The **vision encoder** outputs high-level feature representations from images.
- The **language model** outputs embeddings or feature representations from text.

These representations are inherently different; the visual features are numerical descriptors of visual content, while the language features are numerical descriptors of semantic content. To enable a coherent interaction between these two types of data within LLaVA, the projection matrix transforms the vision encoder's outputs into a space that is compatible with the language model's inputs.

Think of the projection matrix as a translator that converts visual language into linguistic language. By doing this, it ensures that when the model receives an image along with a text instruction, it can combine the information from both sources to generate a response that takes into account both the image content and the textual context.

**Refresher on Projection Matrix Mechanics**

To recall from our LoRA discussion, a projection matrix can be seen as a matrix `W` that, when multiplied by a vector (or matrix of vectors) `X`, yields a new vector (or matrix) `Y` in a new space:

![proj-matrix1.png](images%2Fproj-matrix1.png)


![conceptual-diagram-of-llava.png](images%2Fconceptual-diagram-of-llava.png)

In LLaVA, the projection matrix takes the vision encoder's output `H_v`, and projects it to `Z_v` such that it can be effectively combined with the language model's embeddings `H_q`:

![proj-matrix2.png](images%2Fproj-matrix2.png)

This projection not only aligns the dimensions but also the semantic space, enabling the language model to process and generate a coherent response based on both the visual and textual inputs.

**The Significance of Projection Matrices in Multimodal Learning**

The elegance of using projection matrices in models like LLaVA lies in their simplicity and effectiveness. They offer a parameter-efficient way to bring together different types of data, which is essential for the end-to-end training of multimodal systems. Without projection matrices, it would be challenging to reconcile the disparities between the data types, and the model would struggle to integrate the visual and linguistic information in a meaningful way.

Moreover, projection matrices are learnable, meaning they are fine-tuned during the model's training process to optimize the transformation of visual features for the specific tasks the model is designed to perform. This adaptability is crucial for the model to perform well across a variety of visual and language tasks.

In summary, the projection matrix is a key element that enables LLaVA to perform as a cohesive multimodal system, ensuring that visual and linguistic data are not merely coexisting but are jointly contributing to the model's understanding and output.

### Projection Matrices Made Easy

Grasping the function of projection matrices may seem intimidating initially, particularly when delving into the intricacies of advanced machine learning systems. Yet, this concept is fundamental in the AI landscape, and it merits at least a foundational comprehension of its role and mechanics.

![projection-matrix-made-easy.png](images%2Fprojection-matrix-made-easy.png)

Imagine you're in a dark room with a flashlight. The room represents the data space where we want to understand or find something. Now, the beam of the flashlight is like our projection matrix, and whatever the light touches becomes what we focus on.

In a Transformer model, like someone moving the flashlight around to focus on different objects in the room, the projection matrices shine a light on different parts of the data. When the model projects the input data into queries, keys, and values, it's like adjusting the beam to highlight specific features or details we want to examine more closely. The queries, keys, and values are like different colors of light each showing us something unique:

- The **queries** are like a blue light that seeks out information; they're looking for something specific.
- The **keys** are like a red light that marks where information is; they signal what can be found where.
- The **values** are like a green light that gives us the actual information; they illuminate what we've found.

The model uses these colored lights to decide where to focus its attention, much like you'd use your flashlight to read labels in the dark room to find what you're looking for.

![projection-matrix-made-easy2.png](images%2Fprojection-matrix-made-easy2.png)

In the context of LLaVA, the flashlight—our projection matrix—is being used to merge two different rooms, one filled with visual data like pictures and the other with textual data like words. The projection matrix adjusts the light so that both the image and the text are lit up in a way that they can be understood together. It's like shining a light that can reflect off the picture and cast a shadow of words, merging the two in a way that makes sense to us.

So, in simpler terms, projection matrices in AI models are like beams of light from a flashlight that help the model to "see" and "focus" on the important parts of the data it's trying to understand, whether that's within a single modality like text or across multiple modalities like text and images.

### Projection Matrices in Linear Algebra

In linear algebra, a projection matrix is used to map vectors onto a subspace. This is typically visualized on Cartesian planes where you can see the original vector and its projection onto a line or plane.

Let's consider the 2D Cartesian plane for simplicity. Imagine you have a vector `v` and you want to project it onto a line defined by another vector `u` The projection of `v` onto `u` is a vector that lies on the line of `u`, and it is the "shadow" or "reflection" of `v` when a "light" is shone "down" from a "source" perpendicular to `u`.

Here is how we mathematically determine the projection of `v` onto `u`:

First, we calculate the dot product of `v` and `u`. Then we divide this by the magnitude of `u` squared, which is `∥u∥^2`. This scalar tells us how much of `v` is in the direction of `u`. Finally, we multiply this scalar by the vector `u` to get the projection of `v` onto `u`, denoted as:

![proj-formula1.png](images%2Fproj-formula1.png)

The formula for the projection is:

![proj-formula2.png](images%2Fproj-formula2.png)

To put this into an example with actual vectors:

![proj-formula3.png](images%2Fproj-formula3.png)

1. Calculate the dot product `v ⋅ u`:
 
![proj-formula4.png](images%2Fproj-formula4.png)

2. Calculate the magnitude of `u`squared:

![proj-formula5.png](images%2Fproj-formula5.png)

3. Calculate the projection of `v` onto `u`:

![proj-formula6.png](images%2Fproj-formula6.png)

In this example, the vector `v` has been projected onto the line of `u`, resulting in a new vector:

![proj-formula7.png](images%2Fproj-formula7.png)

It hat lies on the x-axis, which is the direction of `u`.

If you visualize this on a Cartesian plane, you'll see the original vector `v` and a line along the vector `u`, with the projection:

![proj-formula1.png](images%2Fproj-formula1.png)

Lying on that line. This visual representation helps to conceptualize how projection matrices operate in the context of linear transformations.

![vector-projection-seaborn.png](images%2Fvector-projection-seaborn.png)

Here is the graph depicting the vectors `v` and `u`, as well as the projection of `v` onto `u` in the Cartesian plane. The vector `v` is shown in green, `u` in blue, and the projection `proj_u_v` in red.

The graph visually represents the concept of projecting a vector `v` onto another vector ₩ in a two-dimensional space. Here's a detailed explanation of what the graph illustrates:

- **Vector `v` [Green Arrow]**: This vector represents the original vector we wish to project, with components [2,3]. It originates from the origin (0,0) and extends into the Cartesian plane, pointing towards the point (2,3). Keep in mind that vectors are characterized by both direction and magnitude. The vector in discussion is depicted by the line extending from the origin to the point (2,3).

- **Vector `u` [Blue Arrow]**: This vector represents the target of the projection for vector `v`, with its components defined as [4,0]. Positioned exclusively along the x-axis, it signifies its direction. In the graph, the projection appears overlaid, demonstrating how `v` is mapped onto this axis.

![vector-projection-seaborn2.png](images%2Fvector-projection-seaborn2.png)

- **Projection of `v` onto `u` [Red Arrow]**: The red arrow shows the result of projecting `v` onto `u`, resulting in a new vector with components [2,0]. This vector lies on the line defined by `u` and represents the "shadow" or "footprint" that `v` casts onto `u` when a light is shone perpendicularly from `v`down to the line of `u`.

The concept of projection here is akin to dropping a shadow directly downwards (in the direction perpendicular to the line along `u` from `v` to the line defined by `u`. This "shadow" is the part of `v` that aligns with the direction of `u`, effectively "collapsing" `v`'s y-component because `u` has no y-component (it lies entirely on the x-axis).

The visual representation does not imply simply extending one vector or adding vectors in the typical sense but instead shows how `v` is mathematically "flattened" or "projected" onto the direction of `u`, illustrating a fundamental concept in vector projection within linear algebra.

Now, you're ready to dive into the LlaVA papers.

[Deep Dive into LLaVa - Multi-Modal Language Model Part II](README2.md)