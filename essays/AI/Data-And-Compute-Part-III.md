# 💎 Data & Compute Part III 

Common misconceptions, especially if you come from non-computing backgrounds.  

No, GPTs can't literally read.

No, GPTs can't literally see.

No, GPTs can't literally draw.  

No, GPTs can't literally hear.

No, GPTs can't literally talk.

What GPTs or any other AI models do is crunch numbers, represented as vectors or matrices to be exact. Hence the need for vectorized computation. 

These models simulate understanding or perception through mathematical and statistical methods.

Take vision models like GPT-4V for example. It’s essentially still a predictive language model, just one focused on generating descriptions of visual content. A piece of video is a series of images combined sequentially. So to process it, you first need to analyze each image using CNNs (convolutional neural networks) - these are visual detection algorithms that scan images in small grid patterns. The higher the resolution, the more computation is required: a color image has at least 3 channels (RGB) plus an alpha channel (RGBA) for transparency. Even with a simple 3x3 filter, a 1024x1024 resolution image requires processing 1024x1024x3x9 (3 channels x 3 rows x 3 columns) pixel values, with this convolution process repeated across the entire image. More filters require even more computation.  

Vision models can also be trained to associate text descriptions and embeddings with images, similar to systems like Stable Diffusion. With embedded text, models can encode aspects of context and meaning.  

Encoding context also relies heavily on attention mechanisms - this is why transformers are ubiquitous in AI. Attention allows models to selectively focus on relevant parts of their input. Even for multi-modal data like images, video and audio, attention mechanisms can draw connections between elements based on the overall context.  

Image generative models are no different. Stable Diffusion is an image generation model that creates images by starting with random noise, and then modifying that noise through a process called diffusion. The model pays "attention" to certain areas of the noise canvas based on the text prompt provided by the user.

More specifically, Stable Diffusion has an internal representation of the image, called the latent space. This latent space starts out as completely random noise. The text prompt is then encoded into a vector and used to guide the model's attention to relevant areas of the latent noise. The more coherent and specific the prompt, the better the model can focus its attention on pertinent regions to generate the desired image. Thus, the textual prompt provides critical contextual associations for the image being created, allowing narrowing of the latent space by leveraging the power and flexibility of vector embeddings for natural language.

The notion of "attention" in AI refers to focusing computational resources on the most relevant parts of the data, be it text, video, or audio. 

Under the hood, all this computation happens using vectors and matrices crunched on hardware like GPUs that excel at parallelized vector math. This is orders of magnitude faster than non-vectorized operations. That is why there is such heavy focus on GPU performance rather than CPUs for AI workloads - GPUs are designed to handle vectors and tensor (multi-dimensional vector) operations efficiently at scale. Specialized hardware like TPUs (Tensor Processing Units) can be used in certain AI computations.

While systems like GPT don't literally see or hear, everything from sensory data to model weights gets encoded mathematically for huge arrays of vectorized computations.

Even the parameters and weights of AI models fundamentally comprise numbers - typically using float values (e.g. 3.14159...) which represent imprecise real numbers. These get quantized or rounded when run on local hardware due to precision limitations and finite compute resources. Precision is crucial in numerical computations - for example, space rockets can catastrophically fail due to minuscule errors in guidance calculations. Similarly, optimizing precision in deep learning calculations enables more accurate model performance and stability during training and inference. Though conceptually abstract, the essence of data and model representations in AI reduces to carefully constrained real numbers codified in digital hardware.

Don't conclude that AI models resemble human brains. Although initially inspired by neuroscience, modern AI systems have diverged significantly from their origins as brain models. Neural networks actually function quite differently than real biological neurons. Their precise computational workings remain poorly understood — even experts working directly on advanced models admit as much. Furthermore, neuroscience itself still lacks comprehensive understanding of natural brains, not even fully reverse engineering a rat's brain yet. 

GPT-4 has over 1 trillion parameters. We broadly know the algorithms, but cannot trace exactly how responses get generated across those parameters. It just emerges from data and compute. 

Hence why those are the key factors behind AI progress: data and compute.
