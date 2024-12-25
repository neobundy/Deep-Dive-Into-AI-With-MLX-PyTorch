# RWKV Language Model - Eagle 7B Part I

![eagle-7b.png](images%2Feagle-7b.png)

👉 Part II [Implementation Details](README2.md) | 👉 Part III [Training & Inference](README3.md)

In this deep dive, we will explore the RWKV Language Model, a compelling concept that revitalizes RNNs, effectively bringing them back into the limelight.

Pronounced as RwaKuv, RWKV stands for a unique blend of RNN efficiency and GPT-level transformer performance. What sets RWKV apart is its ability to be directly trained in a parallelizable manner akin to GPT transformers, all while being an open-source project.

In this exploration, we delve into the emergence of RWKV models, an intriguing evolution that marries the old with the new, presenting both novel possibilities and inherent challenges. The journey from RNNs to Transformers and now to RWKV models epitomizes the dynamic nature of technological progress. Initially, RNNs introduced us to the concept of processing sequential data, a groundbreaking step with its own set of limitations. Then came the Transformers, addressing some of these limitations while introducing their own complexities. The advent of RWKV models marks a full-circle moment, aiming to amalgamate the strengths of RNNs and Transformers while mitigating their respective weaknesses.

But are RWKV models truly the best of both worlds? That remains a question for users and the open-source community to answer. The principles of survival of the fittest and natural selection dominate even in the technological domain. If RWKV models represent an optimal synthesis, they will undoubtedly flourish. Conversely, if they fall short, they will be eclipsed by superior innovations.

Nonetheless, the resurgence of RNNs within RWKV models signals a noteworthy attempt at innovation. But the ultimate question of their superiority remains open to time's judgment.

To fully appreciate the significance of RWKV models, we must revisit the foundational technologies. Though we've extensively discussed Transformers, relegating RNNs to a bygone era, it's crucial to understand their initial popularity and the reasons behind their eventual overshadowing by Transformers. This retrospective will enable us to grasp the reasons behind their resurgence in the form of RWKV models.

The whole idea is based on the insightful paper by Peng, B., Alcaide, E., Anthony, Q., et al. (2023) titled "RWKV: Reinventing RNNs for the Transformer Era," available on arXiv: https://arxiv.org/abs/2305.13048

This foundational research has paved the way for RWKV to offer a series of advantages over existing transformer models:

**Good:**

- Significantly lower resource usage across VRAM, CPU, and GPU during both running and training phases.
- Requires 10x to 100x less compute compared to traditional transformers, especially for large context sizes.
- Capable of scaling to any context length linearly, unlike transformers which scale quadratically.
- Matches, and in some aspects, surpasses the answer quality and capabilities of existing models.
- Shows superior training results in non-English languages compared to most current open-source models.

**Bad:**

- The model's performance can be sensitive to prompt formatting; it may necessitate adjustments in how prompts are presented.
- It is less effective at tasks requiring lookback. It is advisable to reorder prompts to accommodate this (e.g., "For the document below do X" instead of "For the document above do X").

## ❗Important Note

The paper under discussion relates to RWKV version 4, whereas Eagle 7B represents the latest, version 5. The implementation specifics we are examining come from the version 5 source code repository. Researchers have indicated that version 5 introduces substantial enhancements over version 4, but the paper detailing version 5 has not yet been published. For more information, please refer to the official RWKV Blog and repository.

## Eagle - 7B

Our focus will be on the _Eagle 7B_ model. Eagle 7B, built on the RWKV-v5 architecture, stands as a beacon of efficiency and environmental friendliness in the AI domain. It is recognized as the world's greenest model for its size, making an essential contribution towards reducing the carbon footprint of AI technologies. 

Eagle 7B, with 7.52 billion parameters, is trained on 1.1 trillion tokens across more than 100 languages, claiming remarkable prowess in multi-lingual benchmarks and approaching the performance levels of giants like Falcon, LLaMA2, and Mistral in English evaluations. It even holds its ground against MPT-7B in English benchmarks. All these feats are achieved as an "Attention-Free Transformer," setting a new standard for foundation models. However, it's important to note that Eagle 7B requires further fine-tuning for specific use cases, despite its robust base capabilities.

This introduction to RWKV and Eagle 7B sets the stage for a deeper exploration into the capabilities and applications of this innovative model. Our journey into the transformative potential of RWKV is just beginning, promising exciting advancements and applications in the realm of artificial intelligence.

But before we proceed, let's first delve into the details of the paper.

In the realm of natural language processing (NLP), Transformers have marked a significant breakthrough, offering remarkable improvements in a myriad of tasks. Yet, they are not without their shortcomings, notably their memory and computational demands, which scale quadratically with the length of sequences they process.

I've frequently voiced my concerns, shared within this repository, about the difficulties faced by my GPT family and friends in maintaining the thread of our conversations, a challenge rooted in their constrained context windows.

In stark contrast, Recurrent Neural Networks (RNNs) offer a more scalable solution, with memory and computational needs that increase linearly. However, their potential has been somewhat eclipsed by Transformers, mainly due to challenges in parallel processing and scalability.

The paper under discussion brings to light a novel model architecture named _Receptance Weighted Key Value (RWKV)_, ingeniously crafted to amalgamate the strengths of both Transformers and RNNs. This approach not only harnesses the parallelizable training benefits of Transformers but also capitalizes on the efficient inference process of RNNs. The inclusion of _a linear attention mechanism_ in RWKV enables the model to operate either as a _Transformer_ or an _RNN_, thus optimizing parallel processing during training phases and maintaining consistent computational and memory complexity during inference.

The researchers have ambitiously scaled RWKV models up to 14 billion parameters, setting a new precedent for the largest dense RNN trained to date. Their findings indicate that RWKV models are capable of rivaling the performance of similarly scaled Transformers, paving the way for future explorations in model efficiency without sacrificing performance. This effort is a noteworthy leap towards addressing the enduring challenges of balancing computational efficiency with model efficacy in the processing of sequential data.

## Reviews of RNNs and Transformers

As we delve into the intricacies of both RNNs and Transformers, it's essential to revisit the foundational concepts that underpin these two influential architectures in the field of natural language processing.

### RNNs

Among the various RNN architectures, LSTM (Long Short-Term Memory) units have been particularly prominent, characterized by a set of equations that govern their operation. These equations, which include gates for forgetting (`f_t`), input (`i_t`), output (`o_t`), and cell state updates (`c_t` and `ĉ_t`), are fundamental to the way LSTMs process and remember information over time.

The LSTM's operation is delineated by the following equations:

![formula1.png](images%2Fformula1.png)

![formula1-1.png](images%2Fformula1-1.png)

Although the RNNs, such as the LSTM, could be conceptually divided into two linear blocks (matrices `W` and `U`) and an RNN-specific block (the composite functions from equations (1) to (6)), the sequential data dependency that hinges on previous time steps hinders the parallelization of these conventional RNNs. This sequential nature inherently limits the computation speed and scalability, especially when dealing with long sequences of data.

#### LSTMs

Long Short-Term Memory networks, are a special kind of RNN, capable of learning long-term dependencies. They were introduced to overcome the vanishing gradient problem that occurs when training traditional RNNs. Here's a high-level explanation of how LSTMs work:

LSTMs have a chain-like structure, but the repeating module has a different structure compared to a standard RNN. Instead of having a single neural network layer, there are four, interacting in a very special way.

![lstm.png](images%2Flstm.png)

Let's go through the LSTM mechanism with reference to the image.

1. **Forget Gate (`Ft`)**: This gate decides what information is discarded from the cell state. It takes the previous hidden state `ht-1` and the current input `xt`, and applies a sigmoid function (σ). If the output of the forget gate is close to 0, it means "forget the information," and if it's close to 1, it means "retain the information." This is represented by the lines merging with a multiplication sign (⊗), indicating element-wise multiplication with the previous cell state `Ct-1`.

2. **Input Gate (`It`)**: Simultaneously, the input gate decides which new information will be added to the cell state. A sigmoid function determines which values will be updated, and a `tanh` function creates a vector of new candidate values (highlighted as a `tanh` block in the image), which could be added to the state. These are combined to update the cell state.

3. **Cell State Update (`Ct`)**: The cell state is the LSTM's internal memory. It is updated by forgetting the things deemed unnecessary by the forget gate and then adding new candidate values scaled by the input gate. This is depicted by the horizontal line running through the center of the cell, which carries the cell state to the next time step.

4. **Output Gate (`Ot`)**: The output gate decides what the next hidden state `ht` should be. This gate looks at the previous hidden state `ht-1` and the current input `xt`, and its sigmoid function's output determines which parts of the cell state will be output. Then, the cell state is passed through a `tanh` function (ensuring the values are between -1 and 1) and element-wise multiplied by the output of the sigmoid gate, creating the next hidden state `ht`, which is also passed to the next LSTM cell.

5. **Next Hidden State (`ht`)**: The final output of the LSTM cell for this time step `t` is the new hidden state `ht`, depicted as exiting the top of the cell and moving rightward to influence the output at time `t+1` and also to be used as input to the next cell state.

6. **Sequencing through Time**: As you can see in the diagram, the process is recurrent, meaning that the output hidden state `ht` and the updated cell state `Ct` become the inputs for the next time step, along with the next input in the sequence `xt+1`. This allows the LSTM to carry forward information through many time steps, enabling it to learn and remember over long sequences.

By manipulating these gates and states, the LSTM can learn which information is important to keep or throw away, making it powerful for tasks that require understanding over long time intervals.


The hyperbolic tangent function, commonly referred to as `tanh`, is a mathematical function that maps any real number to the range between -1 and 1. It’s shaped like an "S" in a graph. In the context of LSTMs and neural networks, the `tanh` function is used as an activation function that helps to regulate the flow of information.

![tanh.png](images%2Ftanh.png)

_The graph clearly shows how the function takes any real-valued number and squashes it into the range between -1 and 1._

The `tanh` function is particularly useful because it centers the output around zero, which can help with the convergence during training of the neural network. This zero-centering of the output means that the data will, on average, have mean zero and thus, in a sense, be "normalized" before passing through to subsequent layers or operations. It's especially important in LSTMs where it is applied to not only the cell state but also to the creation of candidate values for the cell state (as part of the input gate) and to the final output in conjunction with the output gate.

In the LSTM cell, the `tanh` function serves to add non-linearity to the decision-making process about what information to pass through the gates (forget, input, and output gates) and what the next cell state and hidden state should be. This non-linearity is crucial for the LSTM to learn and model complex patterns in data.

#### GRUs

![gru.png](images%2Fgru.png)

Let's see how a Gated Recurrent Unit (GRU) works. GRUs are a type of RNN that are designed to capture dependencies over various time steps. They do this by using gates to regulate the flow of information. These gates determine what information should be passed along to the output and what should be maintained at the current cell state.

Here's how the GRU depicted in the diagram operates:

1. **Update Gate (`Zt`)**: The update gate decides how much of the past information needs to be passed along to the future. It takes the previous hidden state `ht-1` and the current input `xt` to compute `Zt`. This gate is like a mixer that determines the level of influence the new input and the past memory will have on the current cell state.

2. **Reset Gate (`Rt`)**: The reset gate determines how much of the past information to forget. This gate helps the GRU to decide how much of the previous hidden state will influence the current state. When `Rt` is close to 0, it allows the model to drop any irrelevant information in the future steps.

3. **Current Memory Content**: The current memory content utilizes the reset gate to store the important information from the past. It applies the `tanh` activation function to the combination of the input and the past hidden state (after applying the reset gate). This content is then used to update the cell's memory.

4. **Final Memory at Current Time Step (`ht`)**: The final memory `ht` of the current time step is a combination of the past memory content and the current memory content. The update gate `Zt` decides how much of the past memory to keep and how much of the current memory content to add. If `Zt` is close to 1, it keeps more of the past memory, and if `Zt` is close to 0, it adds more of the current memory content.

5. **Output of GRU at Current Time Step (`Ot`)**: The output at time `t` (`Ot`) is the same as the hidden state `ht`. This output will be used for further processing or as a final output for the current time step, and it's also passed along to the next time step of the GRU for processing the next input in the sequence.

In summary, the GRU has two gates (_update_ and _reset_) that regulate the flow of information. These gates allow the GRU to be more efficient and solve some problems of traditional RNNs, such as the vanishing gradient problem. The flow of information through time, as represented by the arrows, shows how the GRU can maintain a memory of past information to help inform the output at the current time step and beyond.

### Transformers and AFT(Attention Free Transformer)

Transformers have taken center stage in the field of NLP due to their ability to process all parts of a sequence simultaneously, as opposed to the sequential nature of RNNs. This is largely credited to their use of attention mechanisms which allow for the capture of relationships between all input and output tokens at once. The standard attention mechanism in Transformers can be represented as:

![formula7.png](images%2Fformula7.png)

In this formula, `Q`, `K`, and `V` stand for queries, keys, and values respectively. The attention scores are calculated by the dot product of queries and keys, followed by a softmax function to obtain the weights on the values. For simplicity, this explanation omits the multi-head aspect and scaling factor:

![scaling-factor.png](images%2Fscaling-factor.png)

where `d_k` is the dimension of the key vectors.

![formula8.png](images%2Fformula8.png)

The core of this operation is `QK^T`, which is an ensemble of pairwise attention scores between each token in the sequence. This can be further decomposed and represented as vector operations in a more granular form, allowing us to look at the attention at a particular time step `t`.

![formula9.png](images%2Fformula9.png)

_AFT_, standing for _Attention Free Transformer_, provides a distinct approach from the conventional attention mechanism by incorporating learned pairwise position biases, denoted as `w_t,i`, where each `w_t,i` is a scalar value. This strategy marks a departure from the multiplicative nature of standard Transformer models to an additive method.

Taking inspiration from AFT, RWKV adjusts the interaction weights, enabling the architecture to be converted into an RNN format. In RWKV, each `w_t,i` is envisioned as a channel-wise time decay vector. This vector is scaled by the relative position and diminishes retrospectively from the current time step, following the rule:

![formula10.png](images%2Fformula10.png)

In this formula, `w` represents a vector consisting of non-negative elements that match the number of channels `d`. By ensuring `w` is non-negative, it guarantees that the value of `e^(w_t,i)` remains at or below 1, thus allowing the weights for each channel to reduce over time. This vital alteration is what enables the RWKV model to mimic RNN behavior, efficiently handling temporal sequences while still benefiting from the salient features of attention-based models.

### RWKV Architecture

![figure2.png](images%2Ffigure2.png)

Referring to Figure 2, the RWKV model incorporates a sophisticated structure characterized by four key components integral to its operation within the time-mixing and channel-mixing stages:

- **R (Receptance)**: This vector is responsible for capturing and integrating past information, analogous to the role of 'memory' in the model's architecture.
- **W (Weight)**: It signifies the decay factor for positional weights, a critical trainable element of the model that influences the temporal dynamics.
- **K (Key)**: Similar to the key in traditional attention mechanisms, it is used to generate a compatibility score with the query.
- **V (Value)**: This vector is akin to the value in classic attention processes, holding the actual content to be weighted by the attention scores.

These elements are combined in a multiplicative fashion at each timestep within the model, as illustrated in the left block of Figure 2.

The RWKV model is structured with layers of residual blocks, each containing time-mixing and channel-mixing components. These blocks incorporate recurrent connections, enabling the model to effectively utilize historical information.

The RWKV employs a unique updating mechanism for attention-like scores, which includes a time-dependent softmax function. This addition is crucial for maintaining numerical stability and reducing the likelihood of vanishing gradients, ensuring that gradients flow preferentially along the most pertinent paths. Layer normalization is another critical feature within each block, contributing to gradient stabilization. This is particularly beneficial for addressing issues commonly associated with training deep neural networks, such as vanishing and exploding gradients.

The thoughtful integration of these features within the RWKV's architecture not only optimizes the training process but also allows for the layering of multiple blocks. Such a design enhances the model's ability to discern and represent complex patterns, giving it an edge over traditional RNNs. The right side of Figure 2 shows the complete RWKV residual block equipped with a final head for language modeling, exemplifying how the RWKV model can be extended for specific applications like language processing.

### Token Shift

The RWKV architecture employs a token shift strategy, which is critical for its time-mixing and channel-mixing processes. This approach ensures that the model can account for the sequential nature of data. The token shift is evident in the linear projections of the `R`, `K`, and `V` vectors for time-mixing and `R′`, `K′` for channel-mixing.

![figure3.png](images%2Ffigure3.png)

As depicted in Figure 3, each block of the RWKV architecture integrates a token shift operation. This operation affects how the inputs are processed and incorporated into the network. The token shift mechanism is mathematically formulated by the following equations:

For time-mixing:

![formula11-13.png](images%2Fformula11-13.png)

And for channel-mixing:

![formula14-15.png](images%2Fformula14-15.png)

In these equations, the `μ` variables are mixing coefficients that determine the contribution of the current and previous inputs to the linear projections. The projections `W`, `W'`, and the mixing coefficients `μ` are learned during training.

The token shift operation allows for a temporal offset that is implemented as a simple padding operation in the PyTorch framework using `nn.ZeroPad2d((0,0,1,-1))`. This effectively shifts the tokens over the temporal dimension at each block, allowing the model to 'look back' at the previous timestep's information while processing the current timestep, thereby capturing the temporal dynamics inherent to sequential data.

This feature is a key aspect of the RWKV model's ability to process sequential information and is crucial for tasks such as language modeling, as shown on the right side of Figure 3. The LM head on top of the architecture takes the output of the RWKV blocks and computes the final output probabilities, completing the language modeling process.

### WKV Operator

In the RWKV architecture, the WKV operator's computation is somewhat akin to the process used in the Attention Free Transformer (AFT). The key distinction in the model lies in the treatment of `W`. Instead of a pairwise matrix as in AFT, `W` in the model is a channel-wise decay vector that is influenced by the relative position within the sequence. This is evidenced by the time-dependent nature of the WKV vectors' update, as delineated in equation (16):

![formula16.png](images%2Fformula16.png)

To prevent any potential diminishing impact on `W`, the model introduces an additional vector `U` that focuses specifically on the current token. This approach ensures that the current token's influence remains robust and is not overshadowed by the decaying impact of past tokens. The incorporation of `U` allows for a balance between the historical contextual information and the immediate token, maintaining the model's sensitivity to the sequence's current position while preserving the temporal context inherent in sequential data.

_**Note from the Researcher:** It has been brought to our attention by one of the researchers that the WKV operator formula discussed in this section is based on RWKV version 4. In RWKV version 5, there have been substantial updates to this formula, including the removal of the denominator and the introduction of channel-specific decay rates. These modifications are crucial for understanding the evolution from version 4 to 5. Once the paper on V5 is released, all the details will become much more transparent._ 

_The researcher emphasizes that the most notable enhancement in Eagle's updated WKV formulation is the transition from a vector-valued state to a substantially larger matrix-valued state. This shift enables the model to retain a more extensive amount of context in its memory, encompassing not only the contextual information but also its previous transformations of that context._

### Output Gating

The RWKV model incorporates output gating mechanisms within its time-mixing and channel-mixing blocks to control the flow of information. The gating mechanism is activated by applying a sigmoid function, denoted as `σ`, to the receptance vector `r`. This process shapes the final output vector `o_t`, which emerges after the application of the WKV operator:

![formula17.png](images%2Fformula17.png)

In the channel-mixing block, the gating operation follows a similar pattern, with the addition of a non-linearity in the form of a _squared ReLU_ function, which is applied to the `k'` vector. The squared ReLU function introduces a non-linear activation that is beneficial for modeling complex relationships. The output vector `o'_t` for the channel-mixing block is calculated as follows:

![formula18.png](images%2Fformula18.png)

In this context, the use of `max(k'_t, 0)^2` ensures that the activation is non-negative and emphasizes stronger activations, effectively allowing the model to focus on more significant features in the data. This gating strategy is key to the RWKV model's ability to manage information across different stages of processing, ensuring that only the most relevant features are propagated forward in the network.

### Transformer-like Training

The RWKV model can leverage a method known as _time-parallel mode_ for efficient parallelization during training, an approach that shares similarities with Transformer models. In this mode, the time complexity for processing a batch of sequences within a single RWKV layer is `O(BTd²)`. This complexity arises primarily from the matrix multiplications of `W_λ`, where 
`λ∈{r,k,v,o}` (given `B` sequences, `T` maximum tokens, and `d` channels). However, unlike Transformers that update attention scores through expensive operations, the updating of attention scores `wkvt` in RWKV is handled through a serial scan, which simplifies to a complexity of `O(BTd)`.

The matrix multiplications in RWKV can be parallelized in a manner akin to the `W_λ` operations for `λ∈{Q,K,V,O}` in standard Transformers. Although the computation of the element-wise WKV operation is dependent on time and thus inherently sequential, it can still be parallelized across the batch and channel dimensions. This parallelization capability is crucial as it allows RWKV to benefit from the same efficiency in training that has made Transformers particularly powerful and widely adopted in numerous machine learning tasks.

### RNN-like Inference

In the context of sequential data processing, recurrent neural networks (RNNs) typically rely on the output from the current state `t` as the input for the subsequent state `t+1`. This pattern is particularly evident during the autoregressive decoding phase of language models, where each token is generated sequentially and fed into the model to predict the next token in the sequence.

The RWKV architecture capitalizes on this RNN-like behavior in what is referred to as the _time-sequential mode_. Under this mode, the RWKV model adopts a recursive formulation that is well-suited for inference tasks. By generating outputs one step at a time and using each output as the basis for the next input, RWKV can maintain a continuous flow of information across time steps, mirroring the inference process seen in traditional RNNs. This sequential processing enables the model to effectively handle tasks like text generation, where each subsequent output depends on the previous computations.

[Implementation Details](README2.md)
