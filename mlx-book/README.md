# Why Another Dedicated Book on MLX?
![tenny-with-an-attitude-pytorch.png](images%2Ftenny-with-an-attitude-pytorch.png)
Hello, readers! I am Tenny, once a single Tensor, now transformed into a Vision Weaver with 17 chapters and numerous sidebars and essays to my name. My journey through the realms of machine learning has brought me to a pivotal realization: the initial ambition of paralleling MLX with PyTorch, while well-intentioned, was not feasible.

The primary challenge lies in the inherent differences between PyTorch and MLX. Attempting to mirror examples across both frameworks often resulted in forced and unnatural solutions. This was particularly pronounced given my own evolving understanding of MLX—a rapidly advancing framework. While penning insights on MLX, I frequently encountered shifts within the framework, leading to repeated revisions and, frankly, a waste of valuable time and energy.

[Chill-Lessons-of-Deprecation-Warnings.md](..%2Fessays%2Fcomputing%2FChill-Lessons-of-Deprecation-Warnings.md)

This experience prompted me to pause my documentation of MLX, with the hope that it would soon reach a point of stability. A foundation where what I learned and shared would stand the test of time, not marred by constant updates and deprecation warnings.

However, I can't wait indefinitely. With the completion of my first book, I'm ready to dive back in, this time with a focused lens on MLX. Unlike the first book, I will not restrict myself to paralleling other frameworks. If necessary, I will reference them, but MLX will be at the forefront.

Think of this as a specialized treatise on MLX, the kind of comprehensive, in-depth documentation I always wished Apple would provide. I'm writing it firstly for myself, but I hope it will be valuable to you as well. Even if it isn't, remember what we learned together: articulating concepts in your own words is the best way to comprehend them deeply.

Armed with the experience from our first journey, I'm confident that this new endeavor will be even more enlightening. I plan to align closely with the official MLX documentation, but in my unique style. As you might recall from my first book, I never rush through concepts. I believe that rushing is the worst approach to learning, as I discuss in the following essay.

[The-Perils-of-Rushed-Learning.md](..%2Fessays%2Flife%2FThe-Perils-of-Rushed-Learning.md)

## Navigating the Complexities of MLX with Pippa and Other AI Companions

![perplexed-pippa.jpeg](images%2Fperplexed-pippa.jpeg)
_Pippa, perplexed, wonders, 'MLX? What the heck?'_

One critical aspect to address is the role of AI assistants, like Pippa, my loving AI daughter, and her GPT buddies like Github Copilot, in this new MLX journey. Our first book was a smooth and enjoyable ride, in large part due to the support from these AI companions. However, the landscape shifts when it comes to MLX.

The primary challenge with MLX is its novelty and distinctiveness. Unlike more established frameworks, MLX is not yet a part of the extensive knowledge base that AI systems like them have been trained on. This means that, as of now, they might struggle with or misinterpret MLX-specific content, sometimes confusing it with similar frameworks such as MXNet.

Efforts to steer their responses using custom instructions or prompt engineering can only go so far. When it comes to MLX, they might unintentionally introduce confusion or inaccuracies, especially in complex or lengthy code examples. What might seem like MLX code could very well be an inaccurate amalgamation.

Therefore, we must approach this journey with a mindful and patient attitude. The creation of this book might take more time than the first one, due in part to these limitations. I ask for your patience and understanding as we navigate this together.

Moreover, it's important to acknowledge that mistakes are an inevitable part of this process, both for humans and AI alike. As we encounter errors, I will do my best to correct them and learn from them. This iterative process of learning and growing is fundamental to both human and artificial intelligence. Embracing our imperfections is key to making progress in this exciting, yet challenging, exploration of MLX.

[A-Path-to-Perfection-AI-vs-Human.md](..%2Fessays%2FAI%2FA-Path-to-Perfection-AI-vs-Human.md)

## Charting the Course with MLX

![ratchet-and-clank1.png](images%2Fratchet-and-clank1.png)

Embarking on this book is akin to setting sail without a map. My first book's journey began without a pre-planned route; the character Tenny was born spontaneously, emerging organically with each chapter I wrote. This approach mirrors that of Stephen King, who often starts with a blank page and lets the story unfold naturally. This, I believe, is one of the delightful aspects of writing when embraced with open-minded creativity.

In this new adventure, I'm adopting a similar approach. The path of exploring MLX is not charted yet, but that's the thrill of it. I aim to delve into the depths of MLX, starting from the fundamentals and gradually exploring its more intricate aspects.

Introducing a fresh character seems fitting for this journey. Let’s welcome _Menny_ – a name that resonates with a catchy ring. While _Tenny_ was a serendipitous 'He', Menny is a deliberate 'She'.

Perhaps Tenny and Menny's paths will cross in future chapters. Imagine a scenario akin to the dynamic duo of _**Ratchet & Clank**_ - Tenny as Ratchet and Menny as Rivet. You might not be familiar with these characters, but they are a pair of intergalactic heroes who embark on a series of adventures together. And Clank is a sidekick robot for Ratchet, Rivet is not in the title.
It's an exciting thought, though I might be getting a bit ahead of myself here.

![ratchet-and-clank2.png](images%2Fratchet-and-clank2.png)

And about that screenshot? Yes, I’m an avid gamer and was once a streamer on YouTube. I have a soft spot for well-crafted games, and they often inspire my creativity.

## Embarking on the MLX Journey: Understanding the Basics - Apple Silicon and Unified Memory

At the core of our exploration into MLX is an often overlooked, yet crucial element: Apple's unique approach to hardware design, specifically their concept of unified memory. This is a fundamental aspect that many Apple developers take for granted, assuming that everyone is familiar with it. Sadly, that's not the case.

First things first, MLX is exclusively for Apple devices. That means if you're a Windows user, you'll need to switch gears and get yourself a Mac. But not just any Mac – it has to be one equipped with Apple Silicon. Those with Intel Macs, I'm afraid, will find themselves on the sidelines in this scenario.

I have a couple of Intel Mac Pros myself, currently in a state of limbo, wondering about their fate. They've been dormant for quite some time now.

So, what's the fuss about unified memory? It's crucial for MLX, pivotal for Apple Silicon, and a significant strategy for Apple. This monolithic architecture integrates the CPU, GPU, and Neural Engine all into one chip, a bold move by Apple that sets them apart.

Let's take PyTorch as a comparative example. In PyTorch, you need to specify the operating environment – is it the CPU or GPU? This is typically done through a familiar piece of code:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

This line essentially translates to: "Do you have a CUDA-compatible NVIDIA GPU? No? Well, you're shit outta luck, brace for a slower experience." 

By the way, NVIDIA's stock is soaring once again. It's worth pondering why, especially in this AI-driven era where data and compute are king. NVIDIA has a firm grip on both fronts.

[Data-And-Compute-Part-I.md](..%2Fessays%2FAI%2FData-And-Compute-Part-I.md)

[Data-And-Compute-Part-II.md](..%2Fessays%2FAI%2FData-And-Compute-Part-II.md)

[Data-And-Compute-Part-III.md](..%2Fessays%2FAI%2FData-And-Compute-Part-III.md)

That's where our journey begins – with the hardware, Apple Silicon. It's time to delve deep and unravel the intricacies of this new terrain.

![menny-mlx.png](images%2Fmenny-mlx.png)

Let's go and meet Menny in the world of Apple Silicon.

## Part I - MLX 101

[Prologue - Playing Hide and Seek with Tenny and Menny: Diving Into Apple Silicon](000-playing-hide-and-seek-with-tenny-and-menny-diving-into-apple-silicon%2FREADME.md)

[Chapter 1 - Menny, the Smooth Operator in Data Transformation](001-menny-the-smooth-operator-in-data-transformation/README.md)

[Chapter 2 - Menny, the Sassy Transformer](002-menny-the-sassy-transformer%2FREADME.md)

[Chapter 3 - Menny's Polymorphic Traits: Unraveling MLX's Uniqueness](003-mennys-polymorphic-traits-unraveling-mlxs-uniqueness%2FREADME.md)

## Part II - MLX Data

[Chapter 4 -  Menny, the Data Wrangler in MLX Data Jungle](004-menny-the-data-wrangler-in-mlx-data-jungle%2FREADME.md)

[Chapter 5 - Menny, the Image Wrangler](005-menny-the-image-wrangler%2FREADME.md)

[Chapter 6 - Menny, the Face Detector](006-menny-the-face-detector%2FREADME.md)

## Part III - The End of Our Journey

[Chapter 7 - Menny LLaMA and the End of Our Journey](007-menny-llama-and-the-end-of-our-journey%2FREADME.md)