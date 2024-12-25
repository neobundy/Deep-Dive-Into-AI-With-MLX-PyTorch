# Model Parameters and VRAM Requirements - Can My GPU Handle It?
![apple-silicon.png](images%2Fapple-silicon.png)
Navigating GPU memory limitations is key when dealing with large-scale machine learning models, and this includes working with Apple Silicon. A case in point is the challenge of running sophisticated models like _Mixtral 8x7B_ on the M2 Ultra, which boasts 192GB of RAM. 

At first glance, it appears feasible to accommodate these complex models within the substantial memory of the M2 Ultra, especially considering its unified memory architecture. Yet, in practice, this task can be more demanding than initially expected.

## Memory Requirements Based on Precision

The memory required for storing model parameters varies based on their precision:

- **Full Precision (`float32`)**: Each parameter, stored as a 32-bit floating point, requires 4 bytes of memory.
- **Half Precision (`float16`)**: Each parameter, represented as a 16-bit floating point, occupies 2 bytes of memory.
- **8-bit Quantization (`int8`)**: Storing a parameter as an 8-bit integer takes up 1 byte of memory.
- **4-bit Quantization (`int4`)**: Each parameter, stored as a 4-bit integer, uses only 0.5 bytes of memory.

To calculate total memory requirements, multiply the number of parameters by the bytes per parameter. For example, a 13 billion parameter model exceeds the 24GB VRAM of a GPU like RTX4090 even in half precision (13 * 2GB = 26GB).

## Mixtral 8x7B's Unique Challenges

The Mixtral 8x7B model operates with around 47 billion active parameters, not 56 billion as its name might suggest. This is due to its Sparse Mixture of Experts architecture. For a deeper understanding, see the [Mixtral 8x7B Deep Dive](mixtral-8x7b%2FREADME.md).

Considering the M2 Ultra with 192GB RAM:

1. **Full Precision**: Requires about 188GB for a 47 billion parameter model, which is beyond the M2 Ultra's capacity.
2. **Half Precision**: Requiring approximately 94GB, running such a model on the M2 Ultra is technically possible but comes with the risk of system instability. This is because, in practical terms, the M2 Ultra's GPU memory can effectively utilize about 70-75% of its total capacity, which is around 134GB. While not an official guideline, my experience suggests that adhering to this 70-75% usage threshold of the advertised memory is a prudent approach to maintain system stability and optimal performance.
3. **Quantized Models**: 8-bit quantization reduces the requirement to about 47GB, and 4-bit quantization further lowers it to around 23.5GB.

## System Stability and Model Complexity
![cuda-gpu.png](images%2Fcuda-gpu.png)
Proper memory allocation prevents system inefficiencies and crashes. Moreover, a model's size isn't the only performance determinant. Its architecture and task complexity also significantly influence inference speed. 

The M2 Ultra provides an advantage for running large models, but careful consideration of model precision, architecture, and memory management is key for stable, efficient performance.

Most models on Hugging Face are in half precision. Adapting the precision to align with your VRAM capabilities can be a practical approach. Using the formula mentioned earlier, opting for 8-bit precision can effectively match your VRAM capacity with the model's memory requirements. For instance, a model with 7 billion parameters in 8-bit precision would need 7GB of VRAM. This means such a model could potentially run on an 8GB MacBook Air, where the available VRAM aligns closely with the model's needs. However, for added caution and to ensure smoother operation, it's advisable to opt for 4-bit precision in such scenarios. This approach provides a safer margin by further reducing the memory requirement, enhancing the likelihood of compatibility and stable performance on systems like an 8GB MacBook Air.

## Real-World Application: Microsoft Surface Laptop Studio 2

![microsoft-surface-laptop-studio2.png](images%2Fmicrosoft-surface-laptop-studio2.png)

**Pop Quiz**: Can a Microsoft Surface Laptop Studio 2 with an RTX4060 GPU (8GB VRAM) efficiently run a 7 billion parameter model in half precision?

**Answer**: No. The requirement for such a model is `7 * 2GB` or `14GB`, exceeding the 8GB VRAM. The Surface Laptop Studio 2 is limited for large model machine learning tasks, unless using heavily quantized models. With 7 billion half-precision parameters, the laptop struggles, needing CPU assistance.

I own one, and from my experience, it's quite limited for machine learning purposes. Almost useless. Consider the reason why many vendors shy away from explicitly advertising the VRAM capacity of the RTX4060 in these systems. The screenshot included here is from the vendor where I purchased mine, who are quite transparent about their specifications. They openly state the VRAM capacity, which is not a common practice. Even Microsoft tends to be less forthcoming about this detail; you usually have to delve deeper into the specifications to discover it. 

If you're evaluating laptops with 8GB GPUs for AI projects, it's wise to reconsider. These systems typically lack the capacity for the high demands of such work. Instead, aim for laptops with 24GB GPUs, which represent the upper limit of consumer GPU VRAM currently available. Opting for this higher VRAM will prove beneficial in the long run. While even a 24GB GPU doesn't guarantee the ability to run all the cutting-edge models, it certainly allows you to work with a broader range of them compared to lower VRAM options.

![apple.png](images%2Fapple.png)

In Apple Silicon environments, it's wise to remember that utilizing up to 70-75% of the advertised memory is safest to ensure stability and performance.

Consequently, whenever I purchase new Windows or Apple Silicon devices, I opt for the highest memory configuration on offer. While even 192GB may not meet the requirements for certain complex models, as highlighted in our discussion, it remains the best available option in the current market.