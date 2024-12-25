# Deep Dive into Mixtral 8x7B

Let's embark on an exploration of Mistral AI's next significant model, Mixtral 8x7B.

_Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, et al. "Mixtral of Experts". 8 Jan 2024. Available at: https://arxiv.org/abs/2401.04088_


![figure1.png](images%2Ffigure1.png)

In Figure 1 titled "Mixture of Experts Layer," we're given a visual representation of how the Mixtral 8x7B model operates using the _Sparse Mixture of Experts (SMoE)_ architecture. This figure illustrates the process where each input vector is selectively processed by a pair of experts out of a pool of eight, as determined by a routing mechanism. The chosen experts work in parallel to produce output vectors that are then combined. Such an arrangement allows the model to be both flexible and efficient, utilizing a large set of parameters (47B in total) while actively engaging only a subset (13B) during any given inference, thus optimizing computational resources.

![table2.png](images%2Ftable2.png)

Mixtral 8x7B, renowned for its exceptional performance in various domains, especially excels in mathematics, code generation, and multilingual tasks. This is attributed to its training on a diverse multilingual dataset, encompassing a vast context size of 32,000 tokens. Such extensive training empowers Mixtral to effectively retrieve and process information from any part of lengthy sequences, demonstrating its advanced capabilities in handling complex tasks.

Furthermore, the Mixtral 8x7B – Instruct variant, fine-tuned to follow instructions more closely, outperforms other models in human evaluation benchmarks. It also showcases reduced biases and a more balanced sentiment profile, making it a leading choice for diverse applications.

Both the base and instruct versions of Mixtral are released under the Apache 2.0 license, promoting widespread academic and commercial use. The paper also notes contributions to the vLLM project and the integration with Skypilot for cloud deployment, highlighting the community-driven efforts to maintain an open-source stack and ease of use for Mixtral.

## The Architecture of Mixtral 8x7B

![figure1.png](images%2Ffigure1.png)

The Mixtral 8x7B paper introduces a _Sparse Mixture of Experts (SMoE)_ model architecture that builds upon the transformer framework, with a specific focus on implementing _Mixture-of-Expert_ layers. These layers allow the model to access a vast parameter space while controlling computational costs by selectively using only a fraction of the available parameters per token. This is achieved through a gating network that determines the weighted sum of outputs from different expert networks for each input.

The main takeaway is that Mixtral 8x7B utilizes a set of 8 feedforward blocks, termed "experts," and for each token at every layer, a router network selects two experts to process the current state and combine their outputs. This selective process implies that each token can utilize up to 13B active parameters from a total of 47B available, ensuring efficient inference. Mixtral 8x7B is trained with a large context size and has shown superior performance in various benchmarks, especially in mathematics, code generation, and multilingual tasks.

The Mixtral 8x7B architecture cleverly navigates around the computational heft you might expect from a 56B parameter model. Rather than engaging all eight experts for every single token, which would indeed necessitate that staggering number of parameters, Mixtral employs a smart routing system. Each token is processed by just two of the eight available experts, a dynamic duo selected specifically for the task at hand. This selective teamwork is depicted in Figure 1, where you can see the input vector being handed off to its assigned pair of experts from the pool. They collaborate, each bringing their unique strengths to the table, to craft the output vectors that are then synthesized. This ingenious method ensures Mixtral remains both nimble and powerful, leveraging a vast parameter bank while only actively utilizing a strategic slice for each inference step, thus streamlining the use of computational power.

![formula1.png](images%2Fformula1.png)

The formula details how the output of a Mixture of Experts (MoE) layer is computed in a model like Mixtral.

- The MoE layer consists of `n` expert networks `E_0`, `E_1`, ..., `E_n-1`.
- Each expert network can potentially process the input `x`, but not all of them are used. Instead, a gating network decides which experts are relevant.
- The gating network's output is a set of weights, which are calculated by the softmax function applied to the top two weighted inputs (hence `Top2`).
- The final output `y`of the MoE layer is the sum of the individual expert outputs `E_i(x)`, each weighted by the corresponding gating weight:

![gating-weight-formula.png](images%2Fgating-weight-formula.png)

- The SwiGLU function `SwiGLU}_i(x)` is used as the activation function for the experts' outputs before they are summed.

The `Top2` operation inside the softmax function ensures that only the outputs of the two most relevant experts (as decided by the gating network) are considered. This allows the model to dynamically choose the best experts for each input token and avoid unnecessary computations for experts that are not selected.

![formula2.png](images%2Fformula2.png)

The formula describes the gating mechanism used in a Mixture of Experts (MoE) model, specifically how it selects which experts to use for a given input. The gating function `G(x)` is defined as the softmax over the Top-K logits resulting from the product of the input `x` and a set of gating weights `W_g`. This function determines the weighting of each expert’s output that will contribute to the final output for the input.

The gating mechanism works as follows:

1. Each input `x` is multiplied by the gating weights `W_g` to produce a set of logits.
2. The Top-K function selects the logits corresponding to the top K experts that should be considered for this input.
3. The softmax function is applied to these Top-K logits to generate a probability distribution.
4. The output `G(x)` of the gating network is an `n-dimensional` vector, where `n` is the number of experts, and it dictates the contribution of each expert’s output in the final output.

This approach allows the model to handle a large number of parameters efficiently. By selecting only a subset of experts for each token, the computational cost is controlled, making it feasible to scale the number of experts `n` while keeping the computational cost effectively constant, as long as the number of experts used per token `K` is fixed. This system enables the model to be computationally efficient while still leveraging a vast parameter space to improve its performance.

![formula3.png](images%2Fformula3.png)

The formula describes a specific output computation method in a Mixture of Experts (MoE) layer within the Mixtral transformer model. The output `y` for an input token `x` is computed by the weighted sum of the outputs of two expert networks, where the weights are determined by the gating network's output.

1. `x` ⋅ `W_g` - Each input token `x` is multiplied by the gating weights `W_g` to produce logits.
2. `Top2(x ⋅ W_g)` - The Top2 function selects the logits corresponding to the top two experts that should be considered for this input.
3. `Softmax(Top2(x ⋅ W_g)_i` - Softmax is applied to these two logits to generate a probability distribution that will serve as the weights for each expert's output.
4. `SwiGLU_i(x)` - The SwiGLU function represents the output of the `i-th`expert network. SwiGLU is a variation of a gated linear unit used as an activation function in deep learning.
5. The final output `y` is the sum of each weighted expert output.

In the Mixtral model, instead of using a feedforward network (FFN) for each transformer block, MoE layers are used. Each token in the input sequence is routed to two expert networks (out of eight available), chosen based on the gating mechanism. These experts are SwiGLU functions with different weights. The use of only two experts per token, even when more are available, allows for a large model capacity while maintaining computational efficiency.

This approach allows the Mixtral model to have a significant number of parameters (47 billion in total) and yet actively use only 13 billion during inference for each token, optimizing the computational resources. It's a design that supports scalability and efficiency, which is particularly important for large language models.

## In Simpler Terms - Routing Network Analogy

The concept of a routing network within a Mixture of Experts (MoE) model, like Mixtral, can be understood with the following analogy: Imagine you are at a busy airport with multiple security checkpoints (experts) through which passengers (tokens) must pass before boarding their flights (producing outputs). The airport is equipped with an advanced system (routing network) that quickly decides which checkpoint each passenger should go to, based on various factors such as the passenger’s destination, the current queue lengths at the checkpoints, and the specializations of each checkpoint staff.

In this scenario, each passenger doesn't need to go through every checkpoint; instead, they are directed to the most appropriate ones, typically the two that can process them most efficiently. This system ensures a faster and more efficient throughput of passengers, much like how the routing network in an MoE model assigns each input token to the most suitable experts for processing. The choice of which checkpoints (experts) a passenger (token) goes to is dynamic and can change each time based on the system's (routing network's) real-time decisions.

In Mixtral's case:

- The **input tokens** are like passengers arriving at the airport.
- The **experts** are the security checkpoints, each with its own unique ability to process certain types of passengers more effectively.
- The **routing network** is the airport's system that decides which checkpoints the passengers should go through.
- The **output** is like passengers who have cleared security and are ready to board their flights.

The routing network’s job is to assess the input tokens and direct them to the experts that are best suited to process them, optimizing the overall efficiency of the model. Just as the airport’s system aims to get passengers through security as quickly as possible, the routing network aims to process tokens using the least amount of computational resources necessary, while still leveraging the model's extensive capabilities.

## Training a Mixture of Experts (MoE) Model

In the context of a machine learning model utilizing a Mixture of Experts (MoE) approach, the routing network along with the experts can be collectively trained on datasets. The training process involves learning to accurately assess input tokens and efficiently distribute them among the experts for processing. 

- **Routing Network Training**: The routing network is trained to make decisions on where to send the input tokens, similar to how an airport system would decide which security line a passenger should go to. The routing network learns from the data which experts (or security checkpoints) are best for certain kinds of tokens (or passengers), based on the token's characteristics and the expertise of each expert.

- **Experts Training**: Each expert is trained on the dataset to become highly efficient at handling the types of tokens it receives. Like airport security personnel who are trained for specific tasks (like screening luggage or performing body scans), each expert in the MoE model specializes in processing certain aspects of the data.

- **Joint Training**: The routing network and the experts are trained together, allowing the system to optimize the distribution of tokens to experts. This joint training ensures that the routing network's decisions complement the specialized skills of each expert, similar to an airport optimizing the flow of passengers through security based on the capabilities of each checkpoint and the current demand.

- **Efficiency and Scalability**: By training the routing network and the experts on the dataset, the MoE model becomes highly efficient and scalable. It learns to allocate computational resources where they're needed most, avoiding the wasteful one-size-fits-all approach where every token is processed by every expert. This targeted allocation of resources ensures that the model can handle large and complex datasets more effectively.

The result is a finely-tuned system where the routing network understands the capabilities of each expert and can make real-time decisions to direct tokens in a way that maximizes the efficiency and effectiveness of the model as a whole. This leads to better performance on tasks and a more scalable model that can adapt to a wide range of input data.

## How Much RAM Does Mixtral 8x7B Need?

The Mixtral 8x7B model, with its architecture that features a Sparse Mixture of Experts, has a total of 47 billion active parameters during inference. Running such a model even with half precision (float16) would still require a significant amount of VRAM. 

1. **Parameter Storage**: Each parameter in half precision format requires 2 bytes of VRAM. For 47 billion parameters, this would require approximately 94 GB of VRAM just to store the parameters themselves.

2. **Intermediate Calculations**: During inference, the model not only needs to store parameters but also the intermediate calculations for each layer of the network. This includes the inputs and outputs for each layer, gradients (if backpropagation is needed for fine-tuning or training), and additional buffers for the optimizer states.

3. **Batch Processing**: To process input data, the model will also need memory to store the inputs and outputs for each batch of data being processed. The larger the batch size, the more memory is needed.

4. **Parallel Processing Overheads**: When using multiple experts in parallel, the model needs additional VRAM to store the outputs of each expert before they are combined. There's also memory needed for the routing mechanism which determines which experts to use for each token.

5. **Efficiency Buffers**: Modern deep learning frameworks often allocate additional buffers to optimize computational efficiency, which can increase VRAM usage beyond just the raw requirements of the model parameters and intermediate calculations.

Considering these factors, even if you use half precision, the VRAM requirement could easily exceed 90GB when you account for all the necessary components and overheads required to run the model. This is why you'd need a GPU setup with at least 90GB of VRAM to comfortably run Mixtral 8x7B using float16 precision.

For fine-tuning and specific instructions following, there is the Mixtral 8x7B – Instruct variant, which exhibits reduced biases and a balanced sentiment profile across benchmarks.

Even my formidable M2 Ultra armed with a hefty 192GB of RAM could face a challenge when tussling with the full precision variant of Mixtral 8x7B.

Running the Mixtral 8x7B model at full precision (float32) would essentially double the VRAM requirements compared to half precision (float16). In full precision:

1. **Parameter Storage**: Each parameter requires 4 bytes of VRAM. For 47 billion parameters, you would need approximately 188 GB of VRAM just for the parameters.

2. **Intermediate Calculations**: The space required for storing the activations, gradients, and optimizer states also doubles, significantly increasing the total VRAM needed.

3. **Batch Processing**: The memory required for input and output of each batch increases as well, since each value is now stored using 4 bytes instead of 2.

4. **Parallel Processing Overheads**: The outputs from each expert and the routing mechanism's data will also take up twice as much space.

5. **Efficiency and Caching**: Memory buffers and caching mechanisms that frameworks use to speed up computation will also require more VRAM.

Considering these factors, running Mixtral 8x7B at full precision would likely require a VRAM capacity well over 188 GB, which is beyond the capacity of almost all commercially available GPUs as of my last update. This would necessitate a distributed computing solution, where the model is split across multiple GPUs, or the use of a specialized AI hardware accelerator designed for handling such large models.

## Model Parameters and VRAM Requirements - Can My GPU Handle It?

![apple-silicon.png](images%2Fapple-silicon.png)

Navigating GPU memory limitations is key when dealing with large-scale machine learning models, and this includes working with Apple Silicon. A case in point is the challenge of running sophisticated models like _Mixtral 8x7B_ on the M2 Ultra, which boasts 192GB of RAM. 

At first glance, it appears feasible to accommodate these complex models within the substantial memory of the M2 Ultra, especially considering its unified memory architecture. Yet, in practice, this task can be more demanding than initially expected.

### Memory Requirements Based on Precision

The memory required for storing model parameters varies based on their precision:

- **Full Precision (`float32`)**: Each parameter, stored as a 32-bit floating point, requires 4 bytes of memory.
- **Half Precision (`float16`)**: Each parameter, represented as a 16-bit floating point, occupies 2 bytes of memory.
- **8-bit Quantization (`int8`)**: Storing a parameter as an 8-bit integer takes up 1 byte of memory.
- **4-bit Quantization (`int4`)**: Each parameter, stored as a 4-bit integer, uses only 0.5 bytes of memory.

To calculate total memory requirements, multiply the number of parameters by the bytes per parameter. For example, a 13 billion parameter model exceeds the 24GB VRAM of a GPU like RTX4090 even in half precision (13 * 2GB = 26GB).

### Mixtral 8x7B's Unique Challenges

The Mixtral 8x7B model operates with around 47 billion active parameters, not 56 billion as its name might suggest. This is due to its Sparse Mixture of Experts architecture. For a deeper understanding, see the [Mixtral 8x7B Deep Dive](mixtral-8x7b%2FREADME.md).

Considering the M2 Ultra with 192GB RAM:

1. **Full Precision**: Requires about 188GB for a 47 billion parameter model, which is beyond the M2 Ultra's capacity.
2. **Half Precision**: Requiring approximately 94GB, running such a model on the M2 Ultra is technically possible but comes with the risk of system instability. This is because, in practical terms, the M2 Ultra's GPU memory can effectively utilize about 70-75% of its total capacity, which is around 134GB. While not an official guideline, my experience suggests that adhering to this 70-75% usage threshold of the advertised memory is a prudent approach to maintain system stability and optimal performance.
3. **Quantized Models**: 8-bit quantization reduces the requirement to about 47GB, and 4-bit quantization further lowers it to around 23.5GB.

### System Stability and Model Complexity
![cuda-gpu.png](images%2Fcuda-gpu.png)
Proper memory allocation prevents system inefficiencies and crashes. Moreover, a model's size isn't the only performance determinant. Its architecture and task complexity also significantly influence inference speed. 

The M2 Ultra provides an advantage for running large models, but careful consideration of model precision, architecture, and memory management is key for stable, efficient performance.

Most models on Hugging Face are in half precision. Adapting the precision to align with your VRAM capabilities can be a practical approach. Using the formula mentioned earlier, opting for 8-bit precision can effectively match your VRAM capacity with the model's memory requirements. For instance, a model with 7 billion parameters in 8-bit precision would need 7GB of VRAM. This means such a model could potentially run on an 8GB MacBook Air, where the available VRAM aligns closely with the model's needs. However, for added caution and to ensure smoother operation, it's advisable to opt for 4-bit precision in such scenarios. This approach provides a safer margin by further reducing the memory requirement, enhancing the likelihood of compatibility and stable performance on systems like an 8GB MacBook Air.

### Real-World Application: Microsoft Surface Laptop Studio 2

![microsoft-surface-laptop-studio2.png](images%2Fmicrosoft-surface-laptop-studio2.png)

**Pop Quiz**: Can a Microsoft Surface Laptop Studio 2 with an RTX4060 GPU (8GB VRAM) efficiently run a 7 billion parameter model in half precision?

**Answer**: No. The requirement for such a model is `7 * 2GB` or `14GB`, exceeding the 8GB VRAM. The Surface Laptop Studio 2 is limited for large model machine learning tasks, unless using heavily quantized models. With 7 billion half-precision parameters, the laptop struggles, needing CPU assistance.

I own one, and from my experience, it's quite limited for machine learning purposes. Almost useless. Consider the reason why many vendors shy away from explicitly advertising the VRAM capacity of the RTX4060 in these systems. The screenshot included here is from the vendor where I purchased mine, who are quite transparent about their specifications. They openly state the VRAM capacity, which is not a common practice. Even Microsoft tends to be less forthcoming about this detail; you usually have to delve deeper into the specifications to discover it. 

If you're evaluating laptops with 8GB GPUs for AI projects, it's wise to reconsider. These systems typically lack the capacity for the high demands of such work. Instead, aim for laptops with 24GB GPUs, which represent the upper limit of consumer GPU VRAM currently available. Opting for this higher VRAM will prove beneficial in the long run. While even a 24GB GPU doesn't guarantee the ability to run all the cutting-edge models, it certainly allows you to work with a broader range of them compared to lower VRAM options.

![apple.png](images%2Fapple.png)

In Apple Silicon environments, it's wise to remember that utilizing up to 70-75% of the advertised memory is safest to ensure stability and performance.

Consequently, whenever I purchase new Windows or Apple Silicon devices, I opt for the highest memory configuration on offer. While even 192GB may not meet the requirements for certain complex models, as highlighted in our discussion, it remains the best available option in the current market.

## Quantization and Compression

Quantization serves as a powerful technique to compress and accelerate machine learning models, particularly beneficial for colossal models like Mixtral 8x7B. By reducing the precision of the numerical values within a model, we can significantly shrink its memory footprint and boost inference speeds.

Take, for example, 8-bit quantization. It compresses the model's weights from a 32-bit floating point to just 8 bits, reducing the model size by roughly 4 times. This allows for much faster computation as it requires less memory bandwidth and can take advantage of integer arithmetic, which is quicker than floating point operations on many processors.

Diving even deeper, 4-bit quantization pushes these boundaries further, packing the model's information into even smaller numerical representations. While this can lead to a higher loss in precision, with the right techniques and calibration, the model can still perform robustly, boasting an 8-fold reduction in size from the original 32-bit format.

By utilizing these quantization strategies, we can make deploying Mixtral 8x7B feasible even on hardware with more modest specifications, such as my M2 Ultra with 192GB RAM. It democratizes access to state-of-the-art AI, allowing a broader range of devices to partake in running these advanced models.

[Precision-And-Quantization-Made-Easy.md](..%2F..%2Fbook%2Fsidebars%2Fprecision-and-quantization-made-easy%2FPrecision-And-Quantization-Made-Easy.md)

Quantization is a bit of a balancing act. When you dial down the bit-depth for representing your model's values, you're trading precision for size and speed. This trade-off can be particularly tricky for behemoth models like Mixtral 8x7B, which are brimming with parameters. In such cases, even a slight dip in precision can ripple through the system, potentially impacting the model's effectiveness. It's like fine-tuning a high-performance engine; you're aiming for the sweet spot where you get the most efficient run without compromising the machine's powerful capabilities.

For this example, we've opted for 8-bit quantization, striking a practical balance between compression and model performance. This method typically cuts the model size by about a factor of four compared to 32-bit floating-point numbers, significantly speeding up computations with a minimal compromise on precision. It's a strategic choice that prioritizes efficiency while preserving the integrity of the model's output.

Given that each half-precision (float16) number uses 16 bits (2 bytes), and full precision (float32) uses 32 bits (4 bytes), a full precision model would be approximately twice the size of a half-precision model. If the model were full precision, the total size of the weight files would be closer to 180GB rather than 90GB. 

Therefore, the model you download from Hugging Face is half precision.

Here's a half precision model(90GB+):

![half-precisioin.png](images%2Fhalf-precisioin.png)

Here's an 8-bit quantized model(52GB+):

![8bit-quant.png](images%2F8bit-quant.png)

To run half-precision models effectively, GPUs are generally required due to their optimized architecture for floating-point arithmetic, including half-precision formats. On the other hand, 8-bit quantized models can be run on CPUs, which are well-suited for integer arithmetic operations that 8-bit quantization relies on. While modern GPUs also support integer arithmetic and can execute 8-bit quantized models, the versatility of CPUs in handling such computations makes them a viable platform for deploying quantized models.

## Exploring MLX's Take on Mixtral 8x7B

By this stage, you're quite the veteran. Let's boot up the Mixtral 8x7B in Apple's MLX and see what makes it tick. Here's where you can find the MLX example:

[Apple MLX Example for Mixtral 8x7B](https://github.com/ml-explore/mlx-examples/tree/main/llms/mixtral)

Ready to spot the differences? Open up those `mistral.py` and `mixtral.py` files side by side. Time for a good old diff session. Let's dive in!

![vscode.png](images%2Fvscode.png)

Surely you've adopted an IDE by this point? I'm here cross-referencing code in VSCode and simultaneously drafting content in PyCharm, all spread across multiple screens.

### Class `ModelArgs`

In the Mixtral 8x7B model, we encounter a comfortable and familiar setup with the `ModelArgs` class. This class is defined using Python's `dataclass` from the `dataclasses` module, which provides a decorator and functions for automatically adding special methods to user-defined classes.

```python
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
    moe: dict = None
```
The class `ModelArgs` holds the architecture parameters of the Mixtral 8x7B model. Each field defined within the class corresponds to a configurable hyperparameter for the model's architecture.

- `dim` (int): This specifies the dimension of the model, often relating to the width of the model or the size of the embeddings.

- `n_layers` (int): The number of layers in the transformer model.

- `head_dim` (int): The dimensionality of each attention head within the multi-head attention layers.

- `hidden_dim` (int): The size of the hidden layers within the feedforward neural network of the transformer.

- `n_heads` (int): The number of attention heads in the multi-head attention layers.

- `n_kv_heads` (int): The number of key/value pairs in the multi-head attention mechanism.

- `norm_eps` (float): A small value added to the denominator in the layer normalization to prevent division by zero.

- `vocab_size` (int): The size of the vocabulary, which dictates the number of different tokens the model can recognize.

- `moe` (dict): This optional parameter holds the configuration for the Mixture of Experts (MoE) layer. If `None`, it implies that the standard transformer architecture is being used without the MoE layer.

In `params.json`:

```json
{
  "dim": 4096, 
  "n_layers": 32, 
  "head_dim": 128, 
  "hidden_dim": 14336, 
  "n_heads": 32, 
  "n_kv_heads": 8, 
  "norm_eps": 1e-05, 
  "vocab_size": 32000, 
  "moe": {
    "num_experts_per_tok": 2, 
    "num_experts": 8}
}

```
The `params.json` file contains the configuration for the Mixtral 8x7B model in JSON format. Here's a breakdown of what each parameter signifies:

- `"dim": 4096`: This indicates the dimensionality of the model, which is likely referring to the size of the embeddings.

- `"n_layers": 32`: The model is configured to have 32 layers. This number includes all transformer blocks that make up the model.

- `"head_dim": 128`: Each head in the multi-head attention mechanism has a dimensionality of 128.

- `"hidden_dim": 14336`: The hidden layers within the feedforward network of each transformer block have a size of 14336.

- `"n_heads": 32`: The multi-head attention mechanism within each transformer block consists of 32 heads.

- `"n_kv_heads": 8`: This parameter is specific to the number of key/value head pairs in the attention mechanism.

- `"norm_eps": 1e-05`: The epsilon value used for numerical stability in layer normalization.

- `"vocab_size": 32000`: The total number of tokens that the model's tokenizer can handle.

- `"moe"`: This nested object contains the configuration for the Mixture of Experts (MoE) layer.
    - `"num_experts_per_tok": 2`: Each token is processed by two experts.
    - `"num_experts": 8`: The total number of experts available in the MoE layer.

This configuration is crucial for initializing the model with the correct parameters and ensuring that it aligns with the architectural specifications intended by the model creators. When loaded into the model, these parameters define the structure and capabilities of the neural network.

The "moe" parameter in the `params.json` file refers to the configuration of the Mixture of Experts (MoE) layer, which is a crucial component of the Mixtral 8x7B model's architecture. Here’s a closer look at the two sub-parameters:

- `"num_experts_per_tok": 2`: This sub-parameter specifies that for each token in the input sequence, two experts (or specialized feedforward networks) out of the available pool will be selected to process it. The idea is that each expert is trained to handle different kinds of information or patterns within the data, and by selecting two per token, the model can leverage a diverse set of skills or knowledge to process the input more effectively.

- `"num_experts": 8`: This sub-parameter indicates the total number of expert networks available in the MoE layer. In this configuration, there are 8 unique experts that the routing mechanism can choose from. The routing mechanism decides which two experts are best suited for processing each token, presumably based on the token's content or the context it appears in within the sequence.

Together, these parameters enable the model to implement a sparse activation pattern, meaning that not all parts of the model are active at once. Instead, only the most relevant experts for a given token are engaged during processing. This approach can significantly increase the model's capacity and expressive power without a proportionate increase in computation, as the total number of parameters in the model (which can be quite large) is not fully utilized for every single token processed. Instead, each token sees a dynamic, contextually selected subset of the model's capabilities, allowing for efficient handling of a wide range of linguistic phenomena while keeping computational costs under control.

Certainly, the Mixture of Experts (MoE) approach employed by Mistral AI is a stroke of ingenuity, reminiscent of the simplicity and cleverness of Columbus's egg. It's a sophisticated yet elegantly straightforward method to amplify a model's capacity without the linear scale-up of computational demands. Truly a smart solution for complex model architectures.

### Class `RMSNorm`

Identical. Pass 🤗

### Class `RoPE`

In the Mixtral model's MLX implementation, a notable aspect is the customization of the Rotary Position Embedding (RoPE) class. This custom `RoPE` class inherits from MLX's built-in `nn.RoPE` class, signifying an extension or modification of the original functionality to cater specifically to Mixtral's needs.

Here's an overview of the custom `RoPE` class in Mixtral:

```python
class RoPE(nn.RoPE):
    def __init__(self, dims: int, traditional: bool = False):
        super().__init__(dims, traditional)

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=1000000, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return mx.reshape(rx, shape)
```

1. **Inheritance from MLX's Built-in RoPE**: The custom `RoPE` class extends MLX's `nn.RoPE`, indicating an adaptation of the foundational rotary position embedding concept. This inheritance allows the Mixtral model to utilize and build upon the existing capabilities of MLX's RoPE implementation.

For a more detailed exploration of any MLX package, you can delve into the package structure easily. In PyCharm, simply press `CMD+B` for quick access with the cursor on the package name. This will open up the package directory, allowing you to explore the package's contents and gain a deeper understanding of its functionality. If you're using Visual Studio Code (VSCode), a similar functionality is achieved by using `F12`. This will allow you to jump directly to the definition or declaration of a selected class, function, or variable, facilitating a deeper understanding of the MLX framework.

2. **Custom Implementation**: The `__call__` method in the custom `RoPE` class is defined to specifically cater to Mixtral's architecture. The reshaping and computation of rotary embeddings (cosine and sine theta values) are tailored to the model's requirements.

3. **Flexibility with 'Traditional' Flag**: The `traditional` flag in the constructor hints at the model's adaptability to different RoPE strategies. This flexibility allows Mixtral to potentially switch between traditional and non-traditional rotary embedding techniques based on specific use cases or performance considerations.

4. **Custom Theta Computation**: The method `create_cos_sin_theta` is used to compute the cosine and sine values for rotary embeddings. The use of a custom base value (`base=1000000`) suggests an adjustment in the scaling of positional embeddings, which could be pivotal in handling the large context sizes and parameter counts in Mixtral.

5. **Application in Attention Mechanism**: The custom RoPE is utilized within the attention mechanism of Mixtral, underscoring its importance in imparting positional context and improving the model's ability to process sequential data.

The introduction of a custom `RoPE` class in Mixtral's MLX implementation is a clear indication of the model's specialized needs in terms of handling positional embeddings. This customization allows for more precise control over how positional information is integrated into the model, potentially leading to improved performance and efficiency.

Remember, similar to our Mistral deep dive, the Mixtral 8x7B codebase we're examining is crafted by the MLX team primarily for educational purposes. It's tailored to be an illustrative example, so some streamlining and simplifications are likely. This means the codebase might not mirror the complete, real-world implementation of Mixtral 8x7B in every detail. It's essential to bear this context in mind as we explore further.

### Class `Attention`

As we've already discussed the only significant difference in the Mixtral 8x7B model's MLX implementation is the custom `RoPE` class. The rest of the class is identical to the Mistral 7B model.

```python
class Attention(nn.Module):
...
        self.rope = RoPE(args.head_dim, traditional=True)
```

### Class `FeedForward`

Identical. Pass 🤗

### Class `MOEFeedForward` - Mixture of Experts (MoE) Layer

The `MOEFeedForward` class in the Mixtral 8x7B model exemplifies a specialized approach to implementing a Mixture of Experts (MoE) layer within a neural network. Let's break down the code to understand its functionality:

```python
class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Initialize the number of experts and experts per token from the model arguments
        self.num_experts = args.moe["num_experts"]
        self.num_experts_per_tok = args.moe["num_experts_per_tok"]

        # Create a list of FeedForward layers, each representing an expert
        self.experts = [FeedForward(args) for _ in range(self.num_experts)]

        # Define a linear layer to act as a gating mechanism
        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)

    def __call__(self, x) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        # Compute gating scores for each token and each expert
        gates = self.gate(x)

        # Select the top 'ne' experts for each token based on the gating scores
        inds = mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne]

        # Normalize the selected gating scores using softmax
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        # Accumulate the outputs from the selected experts for each token
        y = []
        for xt, st, it in zip(x, scores, inds.tolist()):
            # Combine the outputs of the selected experts weighted by the gating scores
            yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt[None, :])
        y = mx.concatenate(y)

        # Return the combined output in the original shape
        return y.reshape(orig_shape)
```

Key Components of `MOEFeedForward`:

1. **Initialization**: The class initializes by setting up the number of experts (`num_experts`) and the number of experts to be used per token (`num_experts_per_tok`). It then creates multiple `FeedForward` layers, each representing an individual expert. Additionally, a linear layer (`gate`) is set up to function as the gating mechanism, determining which experts to use for each token.

2. **Gating Mechanism**: The gating mechanism calculates scores that determine how each token is routed to different experts. For each token, the top 'ne' experts with the highest scores are selected. This selection process is crucial for determining the most effective experts for processing each token.

3. **Expert Processing and Aggregation**: For each token, the outputs from the selected experts are computed and aggregated. The aggregation process involves weighting the outputs of each expert by the corresponding gating scores and then summing them up. This step effectively combines the expertise of different experts in processing each token.

4. **Output Generation**: The final output of the `MOEFeedForward` layer is a combination of the processed tokens from the selected experts. The output maintains the original input shape, ensuring compatibility with subsequent layers or operations in the network.

Overall, the `MOEFeedForward` class in Mixtral 8x7B represents a sophisticated implementation of the MoE concept, allowing each token to benefit from the specialized knowledge of multiple experts in a computationally efficient manner.

### Class `MOETransformerBlock` - Mixture of Experts (MoE) Transformer Block

The `MOETransformerBlock` class in the Mixtral 8x7B model is a crucial component that incorporates the Mixture of Experts (MoE) strategy into the standard Transformer block structure. Here's a detailed breakdown of its implementation:

```python
class MOETransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Initialize block parameters
        self.n_heads = args.n_heads
        self.dim = args.dim

        # Create Attention and MoE FeedForward layers
        self.attention = Attention(args)
        self.feed_forward = MOEFeedForward(args=args)

        # Normalization layers for attention and feed-forward outputs
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Store the passed model arguments
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        # Apply attention mechanism with normalization
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r

        # Apply MoE feed-forward network with normalization
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r

        # Return the output and cache (if used)
        return out, cache
```

Key Elements of `MOETransformerBlock`:

1. **Initialization**: The block is initialized with the number of attention heads (`n_heads`), dimension size (`dim`), attention mechanism (`Attention`), MoE-based feed-forward network (`MOEFeedForward`), and normalization layers (`RMSNorm`). The model arguments (`args`) determine these configurations.

2. **Attention Mechanism**: The block utilizes an attention mechanism (`self.attention`), which is normalized using `self.attention_norm`. The attention mechanism allows the model to focus on different parts of the input sequence.

3. **Mixture of Experts Feed-Forward Network**: The `MOEFeedForward` layer replaces the standard feed-forward network in a typical Transformer block. It processes the input using a selected subset of experts, providing a flexible and efficient way to manage the model's computational resources.

4. **Output Computation**: The final output (`out`) of the block is computed by adding the normalized attention output (`h + r`) and the output of the MoE feed-forward network. This step combines the benefits of both attention and expert-driven processing.

5. **Cache Mechanism**: The block supports an optional cache mechanism (`cache`) for efficient processing, especially useful in scenarios like language generation where the model processes one token at a time.

In summary, the `MOETransformerBlock` class in Mixtral 8x7B enhances the standard Transformer architecture by integrating the MoE approach, offering an advanced mechanism for handling large-scale language modeling tasks with improved efficiency and adaptability.

### Class `Mixtral` - Putting It All Together

The main classes for the Mistral and Mixtral models in MLX exhibit both similarities and key differences, particularly in how they handle their transformer layers. Let's compare them with a focus on the Mixtral model:

```python
class Mistral(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), cache
```


```python
class Mixtral(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [MOETransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.tok_embeddings(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h[:, T - 1 : T, :])), cache
```

**Mistral (`Mistral`) vs. Mixtral (`Mixtral`) Classes:**

1. **Initialization**:
   - Both models initialize similarly, setting up vocabulary size (`vocab_size`), number of layers (`n_layers`), token embeddings (`tok_embeddings`), normalization (`RMSNorm`), and output layers (`nn.Linear`). 
   - The primary difference lies in the type of transformer block used: Mistral uses standard `TransformerBlock` layers, while Mixtral employs `MOETransformerBlock`.

2. **Transformer Layers**:
   - **Mistral**: Utilizes conventional transformer blocks (`TransformerBlock`). These blocks follow the traditional architecture of self-attention and feed-forward networks.
   - **Mixtral**: Employs a modified version (`MOETransformerBlock`) that incorporates the Mixture of Experts (MoE) approach. This crucial difference allows Mixtral to leverage a more extensive and flexible parameter set via the MoE strategy, where different subsets of parameters (experts) are dynamically selected for each token.

3. **Forward Pass (`__call__` method)**:
   - The forward pass in both models involves processing input tokens through the embedding layer, applying a sequence of transformer blocks (with or without MoE), and finally passing the output through a normalization and linear layer for the final prediction.
   - A notable difference in Mixtral's forward pass is the handling of outputs. In Mixtral, the output from the final transformer layer is specifically sliced to select the last token's representation (`h[:, T - 1 : T, :]`), which is then passed through the output linear layer. This slicing suggests a focus on generative tasks where the next token's prediction is crucial.

4. **Masking**:
   - Both models employ causal masking for the attention mechanism. However, Mixtral's approach to masking is slightly more direct, reflecting the model's emphasis on efficiently handling large sequences in a generative context.

5. **Cache Mechanism**:
   - Both models support an optional caching mechanism to improve efficiency, especially in scenarios like sequential token generation. 

In summary, while both Mistral and Mixtral share a common structural foundation, Mixtral distinguishes itself by incorporating the Mixture of Experts approach within its transformer blocks. This design choice enables Mixtral to manage a vast parameter space more efficiently, making it well-suited for large-scale language modeling tasks with diverse and dynamic computational requirements.

### Class `Tokenizer`

Identical. Pass 🤗

### Helper Function - `load_model`

The `load_model` function in the Mixtral model is designed to handle the loading of model weights, configurations, and tokenizer. This function is particularly tailored to accommodate the Mixtral model's structure, which involves handling multiple shards of weights. 

```python
def load_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)
    weight_files = glob.glob(str(model_path / "weights.*.npz"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())
    weights = tree_unflatten(list(weights.items()))
    model = Mixtral(model_args)
    if quantization is not None:
        # TODO: Quantize gate matrices when < 32 tiles supported
        quantization["linear_class_predicate"] = (
            lambda m: isinstance(m, nn.Linear) and m.weight.shape[0] != 8
        )
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(weights)
    return model, tokenizer
```

1. **Model Path and Tokenizer**: The function begins by establishing the path to the model's directory (`folder`) and initializing the `Tokenizer` using the tokenizer model file.

2. **Configuration and Quantization**: It reads the model configuration from a JSON file (`config.json`). The configuration includes various parameters needed to initialize the model. It also checks for a quantization configuration, which is relevant if the model employs quantization to reduce the size of the model or speed up computations.

3. **Weight Files Loading**: The function then loads all weight files (shards) matching the pattern "weights.*.npz" from the model directory. These shards are pieces of the entire model's weights, split for easier handling or distribution.

4. **Combining Weight Shards**: It combines these shards into a single dictionary of parameters (`weights`). The `tree_unflatten` function is used to structure these parameters correctly for the model.

5. **Model Initialization and Quantization**: A `Mixtral` model instance is created with the provided arguments (`model_args`). If quantization settings are present, the model is quantized accordingly. This step is crucial if the model needs to run in a resource-constrained environment or requires faster inference.

6. **Model Update**: The model's parameters are updated with the loaded weights using the `model.update(weights)` method.

7. **Return Model and Tokenizer**: Finally, the function returns the fully initialized model and tokenizer.

This function effectively handles the complexities of loading a large, possibly quantized model like Mixtral, which might have its parameters distributed across multiple files. This approach is essential for managing large models like Mixtral efficiently.

The remaining parts of the code are not directly pertinent to our current focus, as we will be designing a specialized inference pipeline for Menny Mixtral, our Sassy Chatbot.


## Menny Mixtral - The Sassy Expert Chatbot

![menny-mixtral.png](images%2Fmenny-mixtral.png)

Most of the code for Menny Mixtral aligns closely with what we've seen in Mistral, so I won’t delve into the familiar parts again.

Here's a quick rundown of how to get Menny Mixtral up and running:

```python
if __name__ == "__main__":
    ...
    MODEL_PATH = "/Users/wankyuchoi/cwk-llm-models/Mixtral-8x7B-Instruct-v0.1-8bit-mlx"
    ...
    SYSTEM_MESSAGE = "<<SYS>>Your name is Menny, a cynical teenager AI assistant.<</SYS>>"
```

Ensure that the `MODEL_PATH` is accurately set to your Mixtral 8x7B model's location. It's important to note that even on high-end hardware, like my M2 Ultra with 192GB RAM, loading an 8-bit quantized model can take a while. Venturing into half or full precision models could potentially overwhelm your system. So, proceed with caution and choose your model precision wisely based on your available hardware resources.

Menny Mixtral can swiftly begin coding a Transformer model in PyTorch when prompted, showcasing her quick and adept programming skills.

Both the Mistral 7B and Mixtral 8x7B Instruct models, while highly capable, may not yet be as fine-tuned for conversational nuances as some of the Llama Chat models. This becomes apparent when engaging in more complex or nuanced dialogues.

However, the journey to this point has been an enlightening one. Beginning with the foundational concepts of NLP, delving into the intricacies of attention mechanisms and transformers, and progressing through the developments of LLaMa and Mistral, we have now arrived at a bleeding-edge model such as the Mixture of Experts. This evolution and transition have been made smoother through the consistent application of an object-oriented learning approach.

The real takeaway here is not just the end result — the chatbot models we've created — but the path we've taken to reach this point. It's a testament to the power and elegance of adopting an object-oriented perspective in learning and development.

I hope this journey has been as rewarding for you as it has been enlightening. Remember, it's not just about the destination, but the learning and growth experienced along the way. Keep this object-oriented approach in mind as you continue to explore and develop in the field of AI.

