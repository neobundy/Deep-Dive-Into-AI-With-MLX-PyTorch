# Deep Dive in Meta AI's JEPA - Joint-Embedding Predictive Architecture

![jepa-title.png](images%2Fjepa-title.png)

## Who is Yann LeCun?

![yann-lecun.png](images%2Fyann-lecun.png)

Yann LeCun is a distinguished French-American computer scientist renowned for his groundbreaking work in machine learning, computer vision, mobile robotics, and computational neuroscience. He was born on July 8, 1960. LeCun is notably recognized for receiving the Turing Award, often referred to as the "Nobel Prize of Computing," which underscores his monumental contributions to the field of artificial intelligence, particularly in the development of convolutional neural networks (CNNs).

LeCun serves as the Silver Professor of the Courant Institute of Mathematical Sciences at New York University and holds the position of Vice President and Chief AI Scientist at Facebook (now Meta), where he directs efforts in AI research. His academic journey led him from France, where he completed his undergraduate education and obtained a Ph.D. in Computer Science from the Université Pierre et Marie Curie (Paris 6), to influential roles in the United States, including his work at Bell Labs and AT&T Labs-Research.

Throughout his career, LeCun has made seminal contributions to the advancement of deep learning and neural networks. His development of the LeNet architecture for handwriting recognition was pioneering in the use of convolutional neural networks, laying the foundation for many modern artificial intelligence systems used for image and speech recognition.

LeCun's influence extends beyond his research contributions. He is a prominent advocate for advancing AI technologies and their application in various fields, emphasizing the importance of understanding the underlying principles of learning in machines. Through his work, LeCun continues to shape the future of artificial intelligence, inspiring generations of researchers and practitioners in the field.

## JEPA in Simple Terms

Yann LeCun, a leading figure in artificial intelligence, has a vision to create AI systems that can learn and reason like animals and humans. In simple terms, he wants to make machines that can understand the world around them in the same intuitive way that people and animals do. Here's how he imagines achieving this:

1. **Learning from Experience**: Just like a child learns to recognize objects by looking at them from different angles and in different situations, LeCun wants AI systems to learn by observing the world. They wouldn't need millions of labeled examples to understand something new; instead, they could learn from a few experiences, just like humans and animals do.

2. **Understanding the World**: LeCun envisions AI that can build internal models of how the world works. This means the AI would have a sort of imagination, allowing it to predict future events or understand the outcome of actions without having to try them out first. For example, even if you've never seen a glass fall and break, you can imagine what would happen based on your understanding of the world.

3. **Reasoning and Making Decisions**: He imagines AI that can make decisions based on its understanding of the world, much like how humans make decisions based on what they know and their past experiences. This could mean anything from navigating a new city based on general knowledge of how cities are laid out to making complex decisions with many moving parts, just like a human would.

To make this a reality, LeCun proposes building AI systems with different modules or parts that work together, each responsible for a different aspect of learning, reasoning, or decision-making. This approach aims to make AI more adaptable, flexible, and capable of understanding the world in a way that's closer to how humans and animals do.

## Yann LeCun on a vision to make AI systems learn and reason like animals and humans

Yann LeCun, as part of Meta AI's Inside the Lab event on February 23, 2022, shared his vision on advancing AI towards human-level capabilities. He highlighted the limitations of current AI systems, such as autonomous driving technologies that, despite extensive training with labeled data and reinforcement learning, still fall short of the human ability to quickly learn and adapt to new tasks like driving.

![lecun1.png](images%2Flecun1.png)

LeCun proposes a shift towards building AI that can learn "world models" — comprehensive internal representations of how the world operates. This approach emphasizes the development of AI systems capable of understanding and interacting with the world in a manner akin to human reasoning, learning through observation and minimal interaction in a largely unsupervised manner. Such AI would possess what is often referred to as "common sense," enabling it to make effective decisions in unfamiliar situations by leveraging a collection of world models to guide its expectations and actions.

![lecun2.png](images%2Flecun2.png)

To achieve this, LeCun suggests an architecture composed of six differentiable modules, including a configurator for executive control, a perception module to interpret sensory data, a world model module for predicting and understanding the world, a cost module to evaluate actions, an actor module for decision making, and a short-term memory module. This architecture aims to allow AI to predict, reason, and plan by learning and using world models in a self-supervised fashion.

![lecun3.png](images%2Flecun3.png)

LeCun also introduces the concept of a Joint Embedding Predictive Architecture (JEPA) to facilitate the learning of these world models. JEPA is designed to capture dependencies between different inputs and generate abstract representations that help the AI system make informed predictions. This approach is pivotal for teaching AI to understand complex, dynamic environments and to perform long-term planning.

![lecun4.png](images%2Flecun4.png)

The training of such a system, LeCun argues, could revolutionize AI by enabling machines to learn from passive observation and interaction, much like human babies do. By predicting outcomes from videos and interacting with the environment, AI could develop a nuanced understanding of the world, paving the way for advanced reasoning and planning capabilities.

LeCun's vision extends beyond current methodologies, focusing on building AI systems that can genuinely understand and interact with the world in a human-like manner. This ambitious goal requires significant research and collaboration within the AI community to overcome numerous challenges, including the development of effective training methods and architectures for world models.

## Six Modules of the JEPA Architecture

Yann LeCun proposes a sophisticated architecture for autonomous intelligence comprising six distinct but interconnected modules, each playing a crucial role in enabling AI to learn, predict, and interact with the world in a manner akin to human cognition:

- **Configurator Module**: Acts as the executive control, preconfiguring other modules (perception, world model, cost, actor) for a specific task. It adjusts their parameters to optimize task execution, essentially guiding the AI on how to approach the task at hand.

- **Perception Module**: Processes sensory signals to estimate the current state of the world. It filters this information to focus only on aspects relevant to the current task, as determined by the configurator module.

- **World Model Module**: The most complex component, serving two primary functions: estimating missing information not captured by perception and predicting future states of the world. This module simulates relevant parts of the world, incorporating the uncertainty inherent in real-world dynamics to generate multiple possible outcomes.

- **Cost Module**: Calculates a single scalar value reflecting the agent's level of discomfort, consisting of an immutable intrinsic cost submodule for immediate discomforts (e.g., damage, violation of behavioral constraints) and a trainable critic submodule for predicting future intrinsic costs. The aim is to minimize this cost over time, driving the AI's basic behavioral motivations and task-specific strategies.

- **Actor Module**: Proposes sequences of actions aimed at minimizing the estimated future cost identified by the cost module. It outputs the first action in the optimal sequence, mirroring classical optimal control strategies.

- **Short-term Memory Module**: Maintains records of the current and predicted world states and their associated costs. This module is essential for tracking the evolving context within which the AI operates and making informed decisions.

Together, these modules constitute a comprehensive framework for building AI systems that can understand and navigate the world with a level of autonomy and reasoning similar to that of humans and animals.

## Human Intelligence and the JEPA Architecture

The architecture proposed by Yann LeCun, featuring six modules, mirrors aspects of human intelligence by organizing complex AI functionalities in a way that is somewhat analogous to how our brains process information, make decisions, and interact with the world. 

- **Configurator Module**: This is like the "planner" in the brain. Just as you might plan your approach before starting a task, this module sets up the AI system by adjusting settings across other modules to optimize how the task will be handled. It decides what the AI needs to focus on, similar to setting a goal or intention in human thought.

- **Perception Module**: Think of this as the senses — sight, hearing, touch, etc. Just like how you perceive the world around you, this module takes in sensory information (from data) and figures out what's happening around the AI system. It filters out irrelevant details, focusing only on what matters for the task at hand, much like how you might tune out background noise to listen to a friend speaking.

- **World Model Module**: This can be likened to imagination or the mental models we construct to understand and predict the world. Humans use past experiences to guess what might happen next in a given situation. Similarly, this module helps the AI predict future events and understand parts of the world it can't directly perceive, preparing it for uncertainties and different possibilities.

- **Cost Module**: Reflecting human emotions and instincts, this module calculates the "discomfort" or risk associated with different actions, aiming to avoid harm and seek beneficial outcomes. It's akin to an inner voice that helps you weigh the consequences of actions based on past experiences and inherent desires, pushing you towards what's perceived as good or away from what's bad.

- **Actor Module**: This is like the decision-maker that translates thoughts into actions. Based on the guidance from the "inner voice" (cost module) and the plan (configurator module), it decides the best course of action to take, similar to how you decide to move your hand away from a hot stove to avoid getting burned.

- **Short-term Memory Module**: Much like human short-term or working memory, this module keeps track of what's currently happening and what the AI has planned or predicted, allowing it to adjust its actions based on new information or changes in the environment.

In essence, these modules together create a system that can plan, perceive, predict, decide, and act in complex and changing environments, much like humans do with their intelligence, but within the realm of artificial intelligence.

## I-JEPA vs. V-JEPA in Simple Terms

Let's break down the exciting developments in Meta AI research by Yann LeCun and team into simpler terms, focusing on two breakthroughs: I-JEPA(2023) and V-JEPA(2024).

### I-JEPA: A New Way to Understand Images

Imagine teaching a computer to understand pictures not by telling it what's in the pictures but by letting it figure things out on its own. That's what I-JEPA does. It's a smart way for computers to learn about images without needing us to give them a lot of instructions. Instead of looking at the whole picture at once, I-JEPA focuses on small parts of the image and tries to guess what other parts might look like. It's like giving a computer a puzzle piece and asking it to guess what the rest of the puzzle looks like based on that one piece.

The cool part is that I-JEPA doesn't just memorize images; it really learns to understand them. It does this by using something called a Vision Transformer, which is good at looking at images in chunks (like breaking a picture into smaller pieces) and figuring out how those pieces relate to each other. This helps the computer to really get what's going on in the image, not just see it as a bunch of pixels.

I-JEPA is smart because it learns from images in a way that's more natural, kind of like how we learn by observing things around us. This makes it better at understanding images and can help computers do tasks like recognizing objects in pictures or even guessing what's in a part of the image that it can't see.

#### 🧐 The Vision Transformer (ViT)

Vision Transformer (ViT) is a groundbreaking approach in the field of computer vision that adapts the transformer architecture—originally developed for natural language processing tasks—for image recognition challenges. Introduced by Google researchers in 2020, ViT marks a departure from conventional convolutional neural networks (CNNs) that have long dominated image analysis tasks.

**Core Concept:**

ViT treats an image as a sequence of fixed-size patches, similar to how words or tokens are treated in text processing. Each patch is flattened, linearly transformed into a higher-dimensional space, and then processed through a standard transformer architecture. This process involves self-attention mechanisms that allow the model to weigh the importance of different patches in relation to one another, enabling it to capture both local and global features within the image.

**Key Features of ViT:**

- **Patch-based Image Processing:** ViT divides images into patches and processes them as sequences, enabling the use of transformer models directly on images.
- **Positional Embeddings:** Similar to NLP tasks, ViT uses positional embeddings to retain the spatial relationship between image patches.
- **Scalability and Efficiency:** ViT demonstrates remarkable scalability, showing increased effectiveness with larger models and datasets. It can be trained on existing large-scale datasets to achieve state-of-the-art performance on image classification tasks.
- **Flexibility:** The architecture is flexible and can be adapted for various vision tasks beyond classification, including object detection and semantic segmentation.

**Impact:**

The introduction of ViT has spurred significant interest in applying transformer models to a wider range of tasks beyond language processing. Its success challenges the prevailing assumption that CNNs are the only viable architecture for image-related tasks and opens up new avenues for research in applying attention-based models to computer vision.

### V-JEPA: Learning from Videos Without Being Told What to Look For

Now, let's talk about V-JEPA, which is like I-JEPA's sibling but for videos. Videos are just images that change over time, right? So, V-JEPA learns by watching lots of videos and trying to understand what's happening in them without anyone having to explain anything. It looks at one part of a video and then tries to guess what's happening in another part.

This is super cool because videos have a lot going on. There are things moving, changing, and interacting, and V-JEPA tries to make sense of all this action without needing any help. It's trained on a huge collection of 2 million videos, learning all by itself from just watching.

The big win with V-JEPA is that it gets really good at understanding both what things look like and how they move. This means it can help with tasks that involve both recognizing objects and understanding actions in videos, all without needing to tweak the model much for different tasks.

### Why It Matters

Both I-JEPA and V-JEPA are about teaching computers to learn on their own from images and videos. This is a big deal because it means computers can get better at understanding the world around us without needing so much help from us. It could lead to smarter AI that can do things like help self-driving cars understand their surroundings or improve security cameras to recognize important events.

In simple terms, I-JEPA and V-JEPA are teaching computers to see and understand like humans do, just by looking at the world around them. This could make AI much more helpful and intuitive in the future.

## Deep Dive - The First Paper on I-JEPA

Yann LeCun et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

🔗 https://arxiv.org/abs/2301.08243

The study introduces a novel approach for learning meaningful image representations without the need for manual data augmentation. Researchers developed the Image-based Joint-Embedding Predictive Architecture (I-JEPA), a non-generative method for self-supervised learning from images. The core principle of I-JEPA involves predicting the representations of various target blocks within an image from a single context block. A key aspect of this method is the use of a masking strategy to guide I-JEPA towards learning semantic representations by focusing on predicting large-scale, semantically rich target blocks using spatially distributed context blocks. The researchers' empirical findings, when applying I-JEPA in conjunction with Vision Transformers, demonstrate its scalability and robust performance across diverse tasks, from linear classification to object counting and depth prediction.

![paper1-figure1.png](images%2Fpaper1-figure1.png)

The researchers in this study delve into self-supervised learning from images, distinguishing between two prevalent approaches: invariance-based and generative methods. They critique the reliance on hand-crafted data augmentations in invariance-based methods for potentially introducing biases detrimental to varied downstream tasks. Drawing inspiration from cognitive learning theories, the study proposes a joint-embedding predictive architecture (I-JEPA) aimed at enhancing the semantic richness of self-supervised representations without additional prior knowledge encoded through image transformations. This architecture endeavors to predict missing information within an image, promoting a deeper understanding of image content through a novel multi-block masking strategy. The study's empirical evaluation showcases I-JEPA's superior performance across multiple tasks, highlighting its scalability and efficiency, particularly in contrast to pixel-reconstruction methods and its adaptability to a wider range of tasks without the need for hand-crafted view augmentations.

![paper1-figure2.png](images%2Fpaper1-figure2.png)

The figure illustrates three types of architectures used in self-supervised learning:

1. **Joint-Embedding Architecture (a)**: This setup is designed to learn similar embeddings for compatible input pairs `(x, y)` and dissimilar embeddings for incompatible ones. The x-encoder and y-encoder generate representations`s_x`and`s_y`, respectively. A discriminator,`D(s_x, s_y)`, assesses whether the pair is compatible.

2. **Generative Architecture (b)**: This method aims to reconstruct a signal `y` from an input `x`. It involves a latent variable `z` that captures additional information needed for reconstruction. The x-encoder processes `x`, and the decoder uses `z` to reconstruct `y`, with the goal of minimizing the difference between the actual `y` and the reconstructed`y_hat`.

3. **Joint-Embedding Predictive Architecture (c)**: This architecture extends the joint-embedding approach by incorporating prediction. The x-encoder and y-encoder create embeddings`s_x`and`s_y`, and a predictor aims to predict`s_y`from`s_x`by possibly utilizing additional (latent) variables `z`. The discriminator,`D(s_hat_y, s_y)`, then evaluates the accuracy of the prediction.

The red boxes highlight the discriminator components in both joint-embedding architectures, which are crucial for assessing the relationship between input pairs.

🧐 _The Joint-Embedding Predictive Architecture (JEPA) in simpler terms can be described as a system that tries to predict one part of the data from another part. Imagine you have a jigsaw puzzle; JEPA would be like trying to guess the shape and image of a missing piece based on the pieces you already have in place. It does this by taking an input, creating a digital "sketch" of it (an embedding), and then using that sketch to predict other related information. This prediction is checked against the actual data to see how close the guess was, and the system learns from it to make better predictions in the future. This architecture is used in AI to help machines learn to understand and process data in a way that is similar to how we naturally make predictions based on what we know._

### Background

In the Background section, the researchers discuss self-supervised learning as a method of representation learning where a system is trained to understand the relationships between its inputs. This is framed within Energy-Based Models (EBMs), where inputs that are not meant to be together are assigned high energy, and compatible inputs are assigned low energy, fitting many existing approaches to self-supervised learning into this model.

🧐 _Energy-Based Models (EBMs) are a class of machine learning models that learn to associate a scalar energy value to every configuration of the variables of interest in a system. The key idea is that the configurations that are more probable or correct according to the training data are assigned lower energy, while less probable or incorrect configurations are given higher energy._ 

_In practice, EBMs are designed to learn a function that takes an input (like an image, a set of sensor readings, etc.) and outputs a single number, the energy, such that the correct or desired inputs result in a low energy and incorrect or undesired inputs result in a high energy. During training, the model's parameters are adjusted to shape this energy landscape accordingly._

_Once trained, the energy values can be used for various tasks like classification, where inputs are assigned to the class with the lowest energy, or generation, where the model seeks to produce new examples that have low energy (and are therefore likely to be similar to the training data). They offer a very flexible framework and have been used for different kinds of data and tasks in machine learning._

The study then describes three architectures within this framework:

1. **Joint-Embedding Architectures**: These architectures learn to produce similar embeddings, or mathematical representations, for inputs that are supposed to be similar (compatible inputs) and different embeddings for dissimilar (incompatible) inputs. They often face the challenge of representation collapse, where the model outputs the same embedding regardless of the input variety. To combat this, various methods like contrastive losses, non-contrastive losses, and clustering-based approaches have been explored.

2. **Generative Architectures**: These involve learning to reconstruct an input signal from a related signal using a decoder network. This network might use additional variables to aid in the reconstruction process. Masking techniques are commonly used to create pairs of signals for this type of architecture, where parts of an image are hidden and the system is trained to fill in the gaps. This approach avoids the issue of representation collapse if the additional variables have less informational capacity than the signal being reconstructed.

3. **Joint-Embedding Predictive Architectures (JEPAs)**: These are similar to Generative Architectures but with a significant distinction — the loss function, which measures prediction accuracy, is applied in the space of embeddings rather than the original input space. This means JEPAs learn to predict the abstract representation of a signal from another signal. Unlike Joint-Embedding Architectures, JEPAs aim for representations that predict each other when given additional information, rather than just seeking invariant representations to hand-crafted data augmentations. However, they still need to address representation collapse, which they do by using an asymmetric design between the different encoders.

![paper1-figure3.png](images%2Fpaper1-figure3.png)

The study's proposed architecture, I-JEPA, exemplifies the Joint-Embedding Predictive Architecture. It uses masking within images and seeks to predict embeddings that are informative of each other, conditioned on additional information. This design aims to avoid the pitfalls of representation collapse by using a differentiated structure between the encoders that process different parts of the input.

### Methodology

![paper1-figure4.png](images%2Fpaper1-figure4.png)

The study introduces the Image-based Joint-Embedding Predictive Architecture (I-JEPA), which is illustrated in Figure 3. The primary aim of this architecture is to predict the representations of various target blocks within an image based on a given context block. The researchers use a Vision Transformer (ViT) as the architecture for the context-encoder, target-encoder, and predictor. A ViT is composed of a stack of transformer layers, each involving a self-attention operation followed by a fully-connected Multilayer Perceptron (MLP). The encoder/predictor architecture used in I-JEPA is reminiscent of generative masked autoencoders (MAE), but a significant distinction is that I-JEPA operates in a non-generative manner, with predictions made in the space of representations.

In the I-JEPA framework, targets are the representations of image blocks. An input image is divided into N non-overlapping patches, which are then passed through the target-encoder to yield a sequence of patch-level representations. For loss calculation, M blocks are sampled from these target representations, and the loss is the average L2 distance between the predicted and target patch-level representations. The loss equation provided by the researchers is:

![paper1-formula1.png](images%2Fpaper1-formula1.png)

This equation represents the average loss across M blocks within an image. Specifically, it calculates the L2 norm (also known as the Euclidean distance) between the predicted representation `s_hat_y(i)` and the actual target representation `s_y(i)` for each of the M blocks. The L2 norm here is the square root of the sum of the squared differences, but since it's being squared again in the equation, it effectively calculates the sum of squared differences directly. The outer sum averages this loss over all M blocks, providing a single scalar value that the training process aims to minimize.

🧐 _The L2 norm, also known as the Euclidean distance, is a measure of the magnitude of vectors. It is the most common way of determining the distance between two points in Euclidean space. In simpler terms, if you think of a vector as a line segment from the origin of a coordinate system to a point, the L2 norm is the length of that line segment._

_Mathematically, for a vector `X`:_

![norm-exp.png](images%2Fnorm-exp.png)

_, the L2 norm is defined as:_

![l2-norm.png](images%2Fl2-norm.png)

_This formula is essentially the Pythagorean theorem extended into n dimensions. When dealing with differences between two points, the L2 norm gives a direct measure of the straight-line distance between them. In the context of loss functions in machine learning, minimizing the L2 norm of the difference between predictions and actual values (as seen in the loss equation) means finding the set of predictions that are closest to the true values in terms of Euclidean distance._

_The L1 norm of a vector, also known as the Manhattan distance or taxicab norm, is the sum of the absolute values of its components. It is a way of measuring distance that considers only the horizontal and vertical paths, similar to how a taxi would drive through a grid of city streets._

_Mathematically, for a vector `X`:_

![norm-exp.png](images%2Fnorm-exp.png)

_, the L1 norm is defined as:_

![l1-norm.png](images%2Fl1-norm.png)

_In contrast to the L2 norm, which measures the shortest direct line between two points, the L1 norm is the sum of the lengths of the paths along the axes of the coordinate system. In machine learning, the L1 norm is often used in loss functions where the goal is to penalize the sum of the absolute differences between predicted and actual values. This can lead to different behavior in optimization, such as promoting sparsity in the solution._

### Related Work

The study's related work section examines the active field of self-supervised learning, particularly focusing on joint-embedding architectures. These architectures involve training encoders to generate similar representations, or embeddings, for different views of the same image. To circumvent issues like representation collapse, where a model may generate indistinguishable representations for distinct inputs, the researchers note that existing studies have employed strategies like explicit regularization or architectural constraints. Specific methods to prevent collapse include interrupting gradient flow within one of the branches of the joint-embedding architecture, employing momentum encoders, or using asymmetrical prediction heads.

Moreover, the study highlights the principles underpinning regularization strategies aimed at preventing collapse. These strategies are often inspired by the InfoMax principle, which promotes the idea that representations should capture maximal information about the inputs while adhering to simplicity constraints. Historically, simplicity has been achieved by encouraging representations to be sparse, low-dimensional, or disentangled, meaning the individual elements of the representation vector should be statistically independent. Contemporary methods continue to follow these principles by incorporating self-supervised loss terms that promote InfoMax alongside simplicity, ensuring that the learned representations are informative yet simple.

Overall, the researchers present a comprehensive overview of the techniques utilized in joint-embedding architectures to prevent the collapse of representations, thereby facilitating the learning of rich and diverse features from the data.

## Image Classification

![paper1-table1-2.png](images%2Fpaper1-table1-2.png)

In the section on Image Classification, the researchers present their findings to establish that I-JEPA is capable of learning advanced image representations without the need for hand-crafted data augmentations. They detail the performance of various image classification tasks utilizing linear probing and partial fine-tuning protocols, specifically on models that have been pre-trained on the ImageNet-1K dataset. For the assessment, all models were trained at a resolution of 224 × 224 pixels unless otherwise specified.

When analyzing the ImageNet-1K linear-evaluation benchmark, the study shows that after self-supervised pre-training, the model weights are kept constant, and a linear classifier is trained atop the full ImageNet-1K training set. The study notes that I-JEPA markedly enhances linear probing performance and does so with less computational resources compared to well-known methods such as Masked Autoencoders (MAE), Context Autoencoders (CAE), and data2vec, all of which also do not depend on extensive hand-crafted data augmentations during pre-training. By capitalizing on the improved efficiency of I-JEPA, the researchers were able to train larger models that surpass the best-performing CAE models while using significantly less computation. They also found that I-JEPA benefits from scale; specifically, a Vision Transformer model with a higher resolution outperforms the best models that use view-invariance approaches despite not relying on hand-crafted data augmentations.

### Local Prediction Tasks

![paper1-table3-4.png](images%2Fpaper1-table3-4.png)

The researchers show that the Image-based Joint-Embedding Predictive Architecture (I-JEPA) not only learns semantic image representations that enhance performance in image classification tasks, but also excels in learning local features. This proficiency is demonstrated in low-level and dense prediction tasks such as object counting and depth prediction, areas where I-JEPA outperforms methods based on view-invariance that typically rely on hand-crafted data augmentations.

The study provides empirical evidence of I-JEPA's capabilities through results obtained using a linear probe on the Clevr dataset. After the encoder has been pretrained, its weights are held constant, and a linear model is added on top to execute specific tasks. On tasks like object counting and depth prediction, I-JEPA not only matches but also significantly exceeds the performance of view-invariance methods such as DINO and iBOT, showcasing its ability to effectively capture and utilize low-level image features acquired during the pretraining phase.

## Scalability

![paper1-table5.png](images%2Fpaper1-table5.png)

The researchers discuss the efficiency and scalability of the Image-based Joint-Embedding Predictive Architecture (I-JEPA) in comparison to previous methods. The study reveals that I-JEPA, when evaluated in semi-supervised settings on 1% of the ImageNet-1K dataset, requires less computational power and still manages to perform strongly without the need for extensive hand-crafted data augmentations. Although I-JEPA introduces some additional computational load due to its operation in representation space, it converges faster—requiring approximately five times fewer iterations—thus providing significant overall computational savings. This efficiency gain is evident when comparing a large-scale I-JEPA model (ViT-H/14) to a smaller model of iBOT (ViT-S/16), with I-JEPA requiring less computational power.

## Predictor Visualizations

The researchers investigate the ability of the I-JEPA's predictor component to accurately capture positional uncertainties within image targets. To conduct this qualitative analysis, the outputs of the predictor are visualized. The researchers' methodology for visualization is designed to be reproducible by the wider research community. This is achieved by freezing the weights of the context-encoder and predictor after pretraining, then using a decoder trained on the RCDM framework to convert the averaged predictions back into pixel space. 

![paper1-figure6.png](images%2Fpaper1-figure6.png)

The resulting visualizations, as displayed in Figure 6, vary based on different random seeds. The attributes that remain consistent across these visualizations are indicative of the information encoded in the averaged predictor outputs. It is observed that the I-JEPA predictor is adept at capturing the position-based uncertainties and is capable of reconstructing high-level object parts with accurate poses, such as the back of a bird or the top of a car, as illustrated by the examples provided.

### Ablations

The researchers delve into a thorough examination of the components within the Image-based Joint-Embedding Predictive Architecture (I-JEPA). They follow the same experimental protocol as outlined earlier and report the results of a linear probe that assesses the pre-trained model's performance on a low-shot version of the ImageNet-1K benchmark.

![paper1-table6.png](images%2Fpaper1-table6.png)

![paper1-table7.png](images%2Fpaper1-table7.png)

Key findings from the study's ablation tests include:

- **Multiblock Masking Strategy**: The researchers test different configurations by altering the scale of the target blocks and the context scale, along with the number of target blocks. Through these experiments, it is determined that predicting multiple large (semantic) target blocks and using a context block that is sufficiently informative and spatially distributed is crucial for effective performance.

- **Masking at the Target-Encoder Output**: A significant design decision within I-JEPA is to mask the output of the target-encoder rather than the input. The study compares two approaches: one where masking is applied at the input and another where masking is at the output. The latter approach, when used during the pre-training of a ViT-H/16 model for 300 epochs, is shown to yield more semantically rich prediction targets and enhance linear probing performance, with the top-1 accuracy being significantly higher at 67.3% compared to 56.1% when masking the input.

- **Predictor Depth**: The study also investigates how the depth of the predictor affects performance by pre-training a ViT-L/16 with either a 6-layer or a 12-layer predictor network for 500 epochs. They find that a deeper predictor enhances the model's performance in low-shot settings.

### Conclusion

In the conclusion of the study, the researchers highlight the introduction of the Image-based Joint-Embedding Predictive Architecture (I-JEPA). This method is touted as being both straightforward and efficient for learning semantic representations of images, all while avoiding the dependency on hand-crafted data augmentations. The study has demonstrated that by operating in the representation space rather than directly reconstructing pixel values, I-JEPA can achieve faster convergence and learn representations with richer semantic meaning. This stands in contrast to methods based on view-invariance, as I-JEPA provides an alternative approach for learning general-purpose representations using joint-embedding architectures without the need for hand-crafted view-based data augmentations.

## Deep Dive - The Second Paper on V-JEPA

Yann LeCun et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video

🔗 https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/

This paper investigates the efficacy of feature prediction as an autonomous method for unsupervised learning from videos, presenting V-JEPA—a suite of vision models developed exclusively around a feature prediction objective. Notably, the training of these models does not leverage pretrained image encoders, text, negative samples, reconstruction techniques, or any form of external supervision. Instead, the models are trained on a collection of 2 million videos sourced from publicly available datasets. The study assesses the models' performance on a variety of downstream image and video tasks, revealing that the approach of learning through video feature prediction yields versatile visual representations. These representations demonstrate strong performance across tasks emphasizing both motion and appearance, without necessitating adjustments to the model's parameters. For instance, without modifying the underlying architecture, the study's most extensive model—a Vision Transformer Hybrid/16 (ViT-H/16) trained exclusively on video data—achieves 81.9% accuracy on Kinetics-400, 72.2% on Something-Something-v2, and 77.9% on ImageNet1K, underscoring the potential of feature prediction in unsupervised learning from video.

![paper2-figure1.png](images%2Fpaper2-figure1.png)

The study revisits the concept of feature prediction as an independent objective for unsupervised visual representation learning from videos. Highlighting the significance of the predictive feature principle, the researchers propose V-JEPA, a novel approach integrating recent advancements such as transformer architectures and masked autoencoding frameworks, without relying on traditional supervision sources like pretrained encoders or human annotations. Trained on a vast dataset of 2 million videos, V-JEPA models demonstrate superior performance on various image and video tasks, showcasing the effectiveness of feature prediction in generating versatile visual representations without model parameter adaptation.

### Related Works

The researchers review the domain of self-supervised learning of visual representations, focusing on joint-embedding architectures. These methods involve training pairs of encoders to produce similar embeddings for multiple perspectives of the same image, aiming to sidestep the potential issue of collapsing solutions by implementing explicit regularization or architectural constraints. The study compares various strategies aimed at collapse prevention, including halting gradient flow in one branch of the joint-embedding model, utilizing a momentum encoder, or adopting an asymmetric prediction head. Theoretical investigations into joint-embedding methods with architectural constraints are also discussed, offering insights into their efficacy in avoiding representation collapse without the need for explicit regularization.

The study also explores regularization-based strategies within joint-embedding architectures that seek to maximize the space occupied by the representations, motivated by the InfoMax principle. This principle advocates for representations to be maximally informative about the inputs while adhering to simplicity constraints. Historically, simplicity was enforced by promoting sparse, low-dimensional, or disentangled representations. Contemporary methods continue to apply these constraints alongside InfoMax through self-supervised loss terms.

Furthermore, the study examines an alternative strand of work focused on learning representations by masking parts of the input and predicting the obscured content. While these approaches, including autoregressive models and denoising autoencoders, have shown scalability, they tend to capture features at a lower semantic level compared to joint-embedding methods.

Recent efforts to merge the strengths of joint-embedding architectures with reconstruction-based approaches are highlighted. These combined approaches aim to balance the learning of global image representations with the benefits of local loss terms for enhanced performance across a broader range of computer vision tasks. The study discusses the framework of contrastive predictive coding as related to these efforts, emphasizing its role in discriminating between overlapping image patch representations to enhance predictability and representation quality.

### Methodology

![paper2-figure2.png](images%2Fpaper2-figure3.png)

The methodology section introduces Video-JEPA (V-JEPA), which leverages feature prediction from videos as an unsupervised learning strategy without relying on conventional supervision sources. The researchers train this model on a vast dataset of 2 million videos, aiming to predict the features of one part of a video based on another. This approach circumvents the need for pretrained image encoders or explicit negative examples, employing instead a self-supervised learning framework that emphasizes the prediction of video features to derive versatile visual representations.

![paper2-figure3.png](images%2Fpaper2-figure3.png)

The study outlines the mathematical foundation of V-JEPA's training objective, detailing four critical equations that underpin the methodology:

1. **Equation 1** addresses the challenge of representation collapse, refining the training objective to ensure diversity in the learned representations:

![paper2-formula1.png](images%2Fpaper2-formula1.png)

where `sg(⋅)` represents a stop-gradient operation, preventing the backpropagation of gradients through its argument, and `Δy` specifies the spatio-temporal position of `y` relative to `x`.

2. **Equation 2** modifies the objective to incorporate an L1 regression, stabilizing the learning process by focusing on median-based predictions rather than mean:

![paper2-formula2.png](images%2Fpaper2-formula2.png)


3. **Equation 3**: The optimal predictor `P^*` for the encoder's output is derived by minimizing the L1 norm between the predicted features `P(E_θ(x))` and the actual features `Y`. This is mathematically expressed as:

![paper2-formula3.png](images%2Fpaper2-formula3.png)

4. **Equation 4**: The gradient of the expected value of the optimal predictor's L1 norm with respect to the encoder parameters `θ` is equivalent to the gradient of the median absolute deviation (MAD) of `Y` given `E_θ(x)`. This relationship is denoted by:

![paper2-formula4.png](images%2Fpaper2-formula4.png)

These equations indicate a methodological shift towards using robust statistics for training, which is likely to make the model's learning more stable against outliers and noise in the data, enhancing the reliability of the video features learned by V-JEPA.

The methodology described by the researchers encapsulates a novel approach to learning from videos by predicting features across different portions of the content. This strategy aims to foster the development of AI models capable of understanding and representing visual information in a manner akin to human perception, leveraging the inherent dynamics and complexities of video data.

### What Matters for Learning Representations from Video?

![paper2-table1-2.png](images%2Fpaper2-table1-2.png)

![paper2-table3.png](images%2Fpaper2-table3.png)

![paper2-table4.png](images%2Fpaper2-table4.png)

The researchers explore several design choices and their contributions to the process of learning visual representations from video. The study examines:

- The comparative impact of using a feature prediction objective against a pixel prediction objective.
- The influence of the pretraining data distribution on the model's performance.
- The effectiveness of different feature pooling strategies in utilizing the model's representations for downstream tasks.
- The role of masking strategies in determining what predictive features should be learned from the video input.

### Comparison with Prior Work

The researchers conduct a series of investigations to assess the effectiveness of feature prediction in learning visual representations from video. 

The study compares V-JEPA with other video-based methods that use pixel prediction as their learning objective, ensuring a similar architectural framework for all the models under comparison.
  
The researchers expand the comparison by removing architectural constraints, allowing them to report on the highest-performing models across different architectures for self-supervised video and image pretraining methods.

The focus shifts to evaluating the label efficiency of V-JEPA in comparison to other self-supervised video pretraining methods, determining how well V-JEPA performs with fewer labels.

### Evaluating the Predictor

The researchers qualitatively examine the performance of the V-JEPA models. These models predict the features of a masked spatio-temporal region within a video based on visible regions, utilizing positional information about the masked areas. To assess the accuracy of these feature-space predictions, the study freezes the pretrained encoder and predictor networks and employs a conditional diffusion decoder designed to translate the V-JEPA predictions into interpretable pixel formats. Notably, this decoder only processes the predicted features for the masked regions and is blind to the visible parts of the video.

The study uses V-JEPA's pretrained models to predict the features of the obscured regions in a masked video, subsequently employing the decoder to convert these predictions into pixel space. Visual outputs from the decoder, as displayed in Figure 6b, are analyzed for various random seeds to determine commonalities that indicate information captured by the predictor.

![paper2-figure6.png](images%2Fpaper2-figure6.png)

Figure 6b's visualizations validate that V-JEPA's feature predictions are not only well-founded but also maintain spatio-temporal coherence with the visible segments of the video. Specifically, these visualizations show that V-JEPA's predictor can accurately handle positional uncertainties and generate diverse visual objects across different locations, all while maintaining consistent motion. Additionally, some visualizations suggest that V-JEPA understands the concept of object permanence, evidenced by the consistent representation of objects even when they are partially obscured.

### Conclusion

In the conclusion section, the researchers encapsulate the essence of their study on Video-based Joint-Embedding Predictive Architecture (V-JEPA). The study delineates V-JEPA as a robust framework for unsupervised learning from video, emphasizing its novel approach that does not require conventional supervision sources like pretrained image encoders or annotated data. Through extensive training on a large dataset of 2 million videos, V-JEPA models showcase the ability to predict video features effectively, leading to versatile visual representations. These representations are adept at handling both motion-centric and appearance-centric tasks, demonstrating strong performance even when the model's parameters are not adapted, which is a testament to the potential and efficacy of feature prediction as a method for learning from video data.

## Personal Notes

It's truly remarkable to witness the generosity and foresight of Meta and Yann LeCun's team in their commitment to open-sourcing their innovations. By sharing their groundbreaking work with the world, they not only accelerate the pace of technological advancement but also democratize access to cutting-edge research. 

This open-source ethos ensures that a wide array of individuals and organizations can build upon these foundations, fostering a collaborative environment where knowledge is freely exchanged and innovation is propelled forward. It's a testament to the belief that collective effort and shared resources can lead to monumental achievements in the field of artificial intelligence and beyond. The impact of such openness is profound, enabling us to harness the full potential of these technologies to address complex challenges and create a future where the benefits of AI are accessible to all.

For more on my take on the benefits of open-source, you can check out my essay:
[Maximizing-Open-Source-Benefits.md](..%2F..%2Fessays%2Finvesting%2FMaximizing-Open-Source-Benefits.md)