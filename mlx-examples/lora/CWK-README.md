✏️If you want to provide feedback, please submit an issue instead of a pull request. I won't be able to merge your requests. Thank you for your understanding.

Notes on Contributions
----------------------
[CONTRIBUTING.md](../CONTRIBUTING.md)

Notes on Pull Requests and Issues
---------------------------------
[NOTES_ON_PULL_REQUESTS_AND_ISSUES.md](../NOTES_ON_PULL_REQUESTS_AND_ISSUES.md)

# Apple MLX LoRA Example: A Comprehensive Unofficial Documentation

I presume you've either reviewed the sidebars mentioned below or possess a thorough grasp of the concepts explained within them:

✍️ Attention Mechanisms and Transformers

[Attention-Is-All-You-Need-For-Now.md](..%2F..%2Fbook%2Fsidebars%2Fattention-is-all-you-need-for-now%2FAttention-Is-All-You-Need-For-Now.md)

✍️ LoRA

[LoRA-Made-Easy.md](..%2F..%2Fbook%2Fsidebars%2Flora-made-easy%2FLoRA-Made-Easy.md)

✍️ Normalization

[Normalization-Made-Easy.md](..%2F..%2Fbook%2Fsidebars%2Fnormalization-made-easy%2FNormalization-Made-Easy.md)

✍️ Precision and Quantization

[Precision-And-Quantization-Made-Easy.md](..%2F..%2Fbook%2Fsidebars%2Fprecision-and-quantization-made-easy%2FPrecision-And-Quantization-Made-Easy.md)

Please note that the original examples from Apple may change over time. Even as I was writing this, the original examples were updated. I will try to keep this guide up to date, but please keep this in mind. 

## convert.py

[convert.py](convert.py)

`convert.py` is used to convert a PyTorch model to a format that can be used with MLX.

The `convert.py` script is a utility designed for converting neural network models from PyTorch to MLX, a framework optimized for Apple silicon. Additionally, it offers an option to quantize the model weights, which can reduce the model size and potentially enhance its efficiency. This script facilitates the transition of models from a PyTorch-based ecosystem to MLX, leveraging the capabilities of Apple's silicon. It is particularly useful for users looking to optimize their deep learning models for efficiency and performance on Apple hardware.

1. **Command-Line Argument Parsing**: 
   - The script utilizes `argparse` to handle command-line inputs, allowing users to specify custom paths and options.
The script you've provided is designed for converting models from PyTorch to MLX, with optional quantization. Here's a detailed analysis of each argument's implications and example scenarios:

   1. `--torch-path`:
      - **Purpose**: Specifies the directory path of the source PyTorch model.
      - **Implication**: This argument is essential for locating and loading the PyTorch model that needs to be converted.
      - **Example Scenario**: If you have a model saved in a directory named "pytorch_model_v1/", you would use `--torch-path pytorch_model_v1/`.

   2. `--mlx-path`:
      - **Purpose**: Defines the destination directory for the converted MLX model.
      - **Implication**: This argument determines where the converted model will be stored, including the tokenizer and weights.
      - **Example Scenario**: To save the converted model in a folder named "converted_model/", use `--mlx-path converted_model/`.

       👉 ex) mistral-7B-v0.1 -> mistral-7B-v0.1-mlx

   3. `-q`, `--quantize`:
      - **Purpose**: A flag to enable quantization of the model during conversion.
      - **Implication**: Activating this flag will quantize the model, potentially reducing its size and improving its execution efficiency.
      - **Example Scenario**: Use `--quantize` if you wish to deploy the model on resource-constrained environments where model size and performance are critical.

   4. `--q-group-size`:
      - Purpose: This argument sets the group size for quantization during model conversion.
      - **Quantization Process**: It determines how the weights are grouped during the quantization process. A smaller group size means that weights are quantized in smaller sets, potentially leading to more fine-grained quantization.
      - **Model Accuracy**: The choice of group size can impact the model's accuracy post-quantization. Smaller groups might retain more information (less lossy), but this can vary depending on the model and the data.
      - **Efficiency vs. Accuracy Trade-off**: A larger group size could lead to more aggressive quantization, potentially improving efficiency (e.g., smaller model size, faster inference) but at the cost of potential accuracy loss.
      - **Efficiency Focus**: If the primary goal is to reduce the model size for deployment in a resource-constrained environment (like mobile devices), a larger group size might be chosen. For instance, `--q-group-size 128` could be used to achieve more significant model size reduction.
      - **Balanced Approach**: In cases where a balance between model size and accuracy is needed, a moderate group size could be chosen, such as `--q-group-size 64`.
   5. `--q-bits`:
      - **Purpose**: This argument specifies the number of bits to use per weight in the quantization process.
      - **Model Size and Performance**: The number of bits per weight directly influences the size of the quantized model. Fewer bits per weight result in a smaller model, which can be advantageous for storage and performance, especially in low-memory environments.
      - **Precision and Accuracy**: A lower bit-width (e.g., 2 or 4 bits) can lead to loss of precision in the weights, which might affect the model's accuracy or lead to degraded performance in certain tasks.
      - **Trade-off Considerations**: Choosing the number of bits involves balancing the need for efficiency (in terms of model size and speed) against the potential impact on model accuracy and effectiveness.
      - **Size-Constrained Deployment**: For deploying a model on edge devices where memory is limited, using a smaller bit-width like `--q-bits 2` might be necessary despite potential compromises in accuracy.
      - **Performance-Sensitive Applications**: In scenarios where performance is critical but some level of accuracy reduction is acceptable, a bit-width of 4 (e.g., `--q-bits 4`) could be a good compromise, offering reduced model size while maintaining reasonable accuracy.

    Both `--q-group-size` and `--q-bits` are integral to the quantization process, each affecting the balance between the efficiency of the converted model and its accuracy. The appropriate values for these parameters can vary significantly based on the specific requirements of the deployment environment and the acceptable trade-offs in model performance.

2. **Directory Initialization and Tokenizer Copying**: 
   - The script ensures the destination directory for the MLX model exists, creating it if necessary.
   - It then copies the tokenizer model from the PyTorch directory to the MLX directory, ensuring that text data is processed consistently in the converted model.
   - 👉 ex) tokenizer.model ->  tokenizer.model

3. **Loading and Converting Weights**: 
   - PyTorch model's weights are loaded from a `.pth` file and converted to a NumPy `.npz` file in the MLX model directory. 
   - Weights are converted to `float16` to optimize memory usage, a key step for handling large models efficiently.

4. **Reading and Updating Configuration**: 
   - The script reads the `params.json` file, which contains the model's configuration. 
   - It removes unnecessary parameters, adds missing ones, and adjusts certain values based on the model's state dictionary. 
   - The updated configuration is then saved in the MLX model directory, ensuring the MLX model retains the architectural characteristics of the original PyTorch model.

5. **Quantization Process (Optional)**: 
   - If enabled, the script quantizes the model's weights, reducing their precision to the specified number of bits. 
   - This process can lead to a smaller model size and faster execution, particularly on hardware optimized for low-precision computations. 
   - Users should be aware of the trade-off between model size and precision, as quantization might affect model performance.

6. **Saving the Converted Model**: 
   - After processing, the script saves the converted (and possibly quantized) weights along with the updated configuration in MLX format. 
   - This final step completes the conversion, making the model ready for use or further development in MLX environments.

    In step 6, the code performs two main operations related to loading and saving model weights.

   1. `state = torch.load(str(torch_path / "consolidated.00.pth"))`: This line is loading a PyTorch model's state dictionary from a file named `consolidated.00.pth` located in the directory specified by `torch_path`. The state dictionary includes the model's weights and biases, stored as tensors.

   2. `np.savez(str(mlx_path / "weights.npz"), **{k: v.to(torch.float16).numpy() for k, v in state.items()})`: This line is saving the model's state dictionary into a NumPy `.npz` file located in the directory specified by `mlx_path`. The state dictionary is first converted to a dictionary of NumPy arrays with half-precision floating-point format (`float16`) to reduce memory usage. The `**` operator is used to unpack the dictionary into keyword arguments, where each key-value pair represents a tensor's name and its corresponding values. The `.npz` format allows saving multiple arrays into a single file in a compressed format.

    The `consolidated.00.pth` and `weights.npz` files both contain the model's weights, but they are in different formats and precisions. The `consolidated.00.pth` file is a PyTorch file that contains the model's weights in the original precision (usually `float32`).

    The `weights.npz` file is a NumPy file that contains the same weights but converted to half-precision floating-point format (`float16`). This conversion is done to reduce the memory usage of the model's weights.

    So, while the actual values of the weights are the same in both files, their storage format and precision are different.

    Refer to `models.py` explanation for what each parameter in `params.json` means in the transformer model architecture.

### Quantization

`quantize(weights, config, args)` : The function is responsible for the quantization of a neural network model. Quantization is the process of reducing the precision of the weights and activations of a model to accelerate inference and reduce model size. It is a common technique used in machine learning to reduce the memory usage and computational cost of a model. It is particularly useful for deploying models on mobile devices with limited resources. For more details, refer to the Precision and Quantization Made Easy sidebar.

1. **Copy the configuration**: 
   It starts by creating a deep copy of the configuration dictionary to avoid altering the original settings.

2. **Loading the model**: 
   Next, it initializes an instance of the `Model` using the given `config` parameters unpacked into `ModelArgs`. This step creates an unoptimized, full-precision version of the model.

3. **Mapping weights to MXNet arrays**:
   It transforms the given `weights` using `tree_map` with `mx.array`, converting Python structures containing numerical values into MX arrays which are suitable for the model.

4. **Updating the model with weights**:
   The weights are then organized using `tree_unflatten`, and the model is updated with these weights.

5. **Quantizing the model**:
   This step applies quantization to the model using the `QuantizedLinear.quantize_module` method. This method quantizes all the linear layers within the Model that satisfy the `linear_class_predicate`. Specifically, the predicate ensures that only the `nn.Linear` layers are quantized, and it excludes the output layer which has the same size as the `vocab_size` specified in the config.

6. **Defining quantization parameters**:
   It sets the quantization parameters `group_size` and `bits` in the `quantized_config` using the values provided in `args`, which specify how the model should be quantized (group size specifies the number of weights to quantize together, and bits determine the precision level).

7. **Flattening and storing the quantized weights**:
   After the quantization, the function extracts the quantized model's parameters using `tree_flatten`, converts them into a dictionary and stores them as `quantized_weights`.

8. **Returning the results**:
   Finally, the function returns the `quantized_weights` alongside the `quantized_config`, ensuring the configuration reflects the quantization that was applied to the model.

#### Utility Functions

The following utility functions are designed to assist in transforming the structure of data trees comprising Python collections such as lists, tuples, and dictionaries, which is particularly useful when dealing with nested parameters or configurations in neural network models.

1. `tree_map(fn, tree, *rest, is_leaf=None)`:
   This function recursively applies a given function `fn` to the leaves of a nested data structure `tree`. The `tree` can contain nested lists, tuples, or dictionaries. If additional structures are provided in `rest`, the function applies `fn` across the leaves of those structures in parallel with `tree`. If an `is_leaf` callable is provided, it's used to determine whether an item should be treated as a leaf or not. If no `is_leaf` is provided, the function defaults to apply `fn` to non-list, non-tuple, non-dict items.

    This function is used to apply a given transformation function (such as quantizing the weights) to each leaf node (e.g., each individual weight or bias) in the nested structure of the model's parameters.

2. `tree_flatten(tree, prefix="", is_leaf=None)`:
   This function converts a nested data structure `tree` into a flat list of key-value tuples. The keys represent the paths to the values using dot notation. If the `is_leaf` function is supplied, it's used to determine if an object is a leaf. If it's not supplied, the function treats any object that is not a list or dict as a leaf. This allows for easy serialization or iteration over model parameters which could be deeply nested.

    This function transforms the nested parameter structure into a flat list of key-value pairs, which could be helpful when saving quantized parameters to disk or processing them sequentially.

3. `tree_unflatten(tree)`:
   The inverse operation of `tree_flatten`, this function takes a flat representation of a tree (a list of key-value tuples where the key uses dot notation to indicate the path) and reconstructs the original nested data structure. The data structure is inferred from the keys, so a key like "0.1.2" indicates that the structure should be a list, whereas keys with non-numeric prefixes indicate that the structure should be a dictionary. This function is useful for restoring the structure of parsed or modified configurations or model parameters for further processing or use.

    This function reconstructs the original nested parameter structure from the flat list of key-value pairs, allowing the model to be rebuilt with its new quantized parameters in the correct hierarchical format.

In essence, during quantization, these functions help to navigate, apply transformations, and manage the complex hierarchies of model parameters. It is a straightforward process of quantizing the model's weights. Refer to the Precision and Quantization Made Easy sidebar for more details on quantization.

## models.py

[models.py](models.py)

This file contains the transformer model used in the example. It also contains the `LoRALinear` class, which is a drop-in replacement for a regular `nn.Linear` layer but with the additional LoRA low-rank updates incorporated. 

The script is designed to be flexible, allowing for adjustments in model size, complexity, and functionality through the `ModelArgs` configuration. Each component is designed to work in tandem, providing a cohesive workflow typical of transformer models used in modern natural language processing tasks. The usage of LoRA and RMS normalization are notable features, enhancing the model's adaptability and training stability. The model's design is rooted in the transformer architecture, optimized for performance and efficiency, particularly on Apple's silicon through the MLX framework.

1. **ModelArgs (Dataclass)**
   - A data class used to store model configuration parameters such as dimensions, number of layers, head dimensions, and vocabulary size.
   - `dim`: This is the dimensionality of the model. In the context of transformers, this is usually the dimensionality of the embeddings.
   - `n_layers`: This is the number of layers in the transformer model.
   - `head_dim`: This is the dimensionality of each head in the multi-head attention mechanism of the transformer.
   - `hidden_dim`: This is the dimensionality of the hidden layer in the feed-forward network of the transformer.
   - `n_heads`: This is the number of heads in the multi-head attention mechanism of the transformer.
   - `n_kv_heads`: This is the number of key/value heads in the multi-head attention mechanism of the transformer.
   - `norm_eps`: This is the epsilon value used for normalization in the transformer model to avoid division by zero.
   - `vocab_size`: This is the size of the vocabulary, i.e., the number of unique words in the dataset. This is used to create the word embeddings in the transformer model.

2. **LoRALinear (Class)**
   - A custom linear layer with Low-Rank Adaptation (LoRA). It adapts a standard linear layer (`nn.Linear`) to incorporate low-rank matrices (`lora_a` and `lora_b`), allowing more efficient training and adaptation.
   - The `from_linear` static method allows the conversion of a standard linear layer to a `LoRALinear` layer.

3. **RMSNorm (Class)**
   - Implements Root Mean Square Layer Normalization. It normalizes input features by their root mean square value, improving training stability.
   - The `_norm` private method computes the RMS normalization.

    For more details on the general concept of normalization, refer to the Normalization Made Easy sidebar.

4. **Attention (Class)**
   - Implements the attention mechanism of the transformer. It uses query, key, and value projections (`wq`, `wk`, `wv`) to compute attention scores and produces the output through an output projection (`wo`).
   - The `rope` attribute applies Rotary Positional Embeddings (RoPE) to incorporate positional information.
   - The `__call__` method handles the attention computation, including optional caching for efficient sequential processing.

    For more details on the general concepts of attention mechanisms, positional encodings and transformers in general, refer to the Attention Is All You Need For Now sidebar.

5. **FeedForward (Class)**
   - Represents the feed-forward network within a transformer block. It consists of two linear transformations with a SiLU activation in between.
   - The `__call__` method defines the forward pass through the feed-forward network.

6. **TransformerBlock (Class)**
   - A single transformer block containing an attention layer, a feed-forward network, and layer normalization for both.
   - The `__call__` method describes the forward pass through the transformer block, including residual connections and normalization.

7. **Model (Class)**
   - The main model class assembling the transformer blocks into a complete model.
   - It includes token embeddings (`tok_embeddings`), multiple transformer blocks (`layers`), final layer normalization (`norm`), and an output projection (`output`).
   - The `__call__` method defines the forward pass for the entire model, handling input embedding, optional masking, layer-by-layer processing, and final output generation.

## lora.py

[lora.py](lora.py)

This is a main script designed for finetuning transformer models using LoRA (Low-Rank Adaptation) or QLoRA (Quantized LoRA) techniques. It provides functionalities for training, evaluating, and generating text using these models.

QLoRA, or Quantized Low-Rank Adaptation, is an advanced technique in the field of deep learning, particularly in the context of fine-tuning large pre-trained language models. Here's a brief recap of LoRA and Quantization before we dive into QLoRA. Fore more details, refer to the LoRA Made Easy and Precision and Quantization Made Easy sidebars.

LoRA is a technique designed to fine-tune large language models. These models have hundreds of millions to billions, even trillions of parameters, making them resource-intensive to train and fine-tune. LoRA works by introducing low-rank matrices into the existing pre-trained weight matrices of the model. Instead of updating the entire weight matrix, LoRA updates only these smaller, low-rank matrices. This approach significantly reduces the number of parameters that need to be trained, making the process more computationally efficient.

![paper.png](..%2F..%2Fbook%2Fsidebars%2Flora-made-easy%2Fpaper.png)

In practice, for a weight matrix `W` in a transformer model, LoRA adds two smaller matrices `A` and `B`, such that the update to `W` can be represented as `ΔW = AB^T`. Here, `A` and `B` are much smaller in size compared to `W`, thereby reducing the computational burden.

In `models.py`:

```python
class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self, input_dims: int, output_dims: int, lora_rank: int = 8, bias: bool = False
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        return y + 2.0 * z
```

The `LoRALinear` class definition is an implementation of the same LoRA concept in the paper. It is designed to work as a drop-in replacement for a regular `nn.Linear` layer but with the additional LoRA low-rank updates incorporated. 

1. **Replacement of Standard Linear Layer**: The class has a static method `from_linear` which takes a standard `nn.Linear` layer and a rank as input and outputs a `LoRALinear` object. This allows for easy substitution of an MLX linear layer with its LoRA-enhanced counterpart.

2. **Initialization (`__init__` method)**: The constructor of the `LoRALinear` class initializes both the standard weight matrix `W` of a linear layer and two low-rank matrices `A` (`lora_a`) and `B` (`lora_b`). Note that in LoRA, `W` corresponds to the original, frozen weights (`W0`), and `A` and `B` correspond to the trainable parameters that capture the updates (`ΔW`).

3. **Low-Rank Matrices Initialization**: The low-rank matrices `A` and `B` are initialized with a certain strategy:
   
   - `self.lora_a` is initialized with values from a uniform distribution scaled by the input dimension, which is a common initialization strategy to maintain the variance of activations.
   
   - `self.lora_b` is initialized to all zeros, meaning initially there is no update from the low-rank component (`ΔW` initially is zero).

4. **Forward Pass (`__call__` method)**: The modified forward pass first calculates the normal output of a linear layer `y` and then computes the output `z` of the low-rank structure by applying `x` to `lora_a` and then `lora_b`. The final output of the layer is the sum of `y` and twice the value of `z`, which reflects the LoRA update.

This particular implementation illustrates how the core concepts of LoRA—low-rank factors and efficient modeling of weight updates—can be imbedded directly into neural network architectures using standard machine learning frameworks like MLX and PyTorch.

Quantization in deep learning involves converting a model's weights and activations from floating-point representation (like 32-bit floats) to a lower bit-width format (like 8-bit integers). This process can significantly reduce the model's memory footprint and computational requirements. Quantization can lead to models that are smaller in size and faster in execution, making them more suitable for deployment in resource-constrained environments like mobile devices. A key challenge with quantization is maintaining the performance of the model. Reducing the precision of weights and activations can lead to a loss in accuracy.

QLoRA combines the principles of LoRA and quantization. It adapts the weights of a large pre-trained model using low-rank matrices, and these updates are quantized to reduce precision and computational demands. By applying both techniques, QLoRA aims to make the fine-tuning process more efficient in terms of computational resources and memory usage while attempting to retain the performance benefits of the original large model. This approach is particularly beneficial in scenarios where there are limitations on computational resources, such as edge computing or mobile applications, and where fine-tuning large models is necessary for tasks like language understanding, translation, or text generation.

The challenge with QLoRA lies in balancing the trade-offs between efficiency (due to quantization and low-rank updates) and the performance of the fine-tuned model. It requires careful tuning of the quantization parameters and the size of the low-rank matrices to ensure that the model remains effective after fine-tuning.

### 1. Argument Parsing
- **Purpose**: Handles command-line arguments for model paths, generation, training, and evaluation settings.
- **Key Arguments**:
  - `--model`: Path to model files (tokenizer, weights, config).
  - Generation-related arguments (`--num-tokens`, `--write-every`, `--temp`, `--prompt`).
  - Training-related arguments (`--train`, `--data`, `--lora-layers`, `--batch-size`, etc.).
  - Evaluation and testing arguments (`--test`, `--test-batches`, `--seed`).

### 2. Tokenizer Class
- **Functionality**: Handles tokenization and detokenization using SentencePiece. SentencePiece is an opensource library and toolkit developed for natural language processing tasks, specifically designed to handle text tokenization and detokenization in a way that's highly effective for a variety of languages, including those with no clear word boundaries like Japanese or Chinese.
- **Methods**: Includes methods like `encode` for converting strings to token IDs and `decode` for the reverse process.

### 3. Dataset Class
- **Description**: A lightweight wrapper for handling data from JSONL files.
- **Usage**: Facilitates data loading for training, validation, and testing.

### 4. Model Loading and Configuration
- **Model Loading**: Loads the pre-trained model and tokenizer from specified paths.
- **Configuration**: Reads and applies model configuration from JSON files.

### 5. Training and Loss Computation
- **Loss Function**: Defines a function to compute the loss for training.
- **Training Loop**: Includes the main loop for training the model, with gradient updates and performance reporting.

### 6. Batch Iteration
- **Function**: Iterates over batches of data for training or evaluation.
- **Details**: Handles tokenization, padding, and batch creation.

### 7. Model Evaluation
- **Purpose**: Evaluates the model on validation or test datasets.
- **Implementation**: Computes average loss over batches of data.

### 8. Text Generation
- **Functionality**: Generates text based on a given prompt using the trained model.
- **Details**: Includes sampling strategies and temperature-controlled generation.

### 9. Main Execution Flow
- **Initial Setup**: Parses arguments and sets up the model and tokenizer.
- **Model Adaptation**: Adapts certain layers (e.g., attention weights) using LoRA.
- **Training Execution**: Conditional execution of training, adapter loading, and weight saving.
- **Evaluation and Testing**: Conditional execution of model evaluation on test datasets.
- **Text Generation**: Conditional execution of text generation if a prompt is provided.

### Functions

Each function is designed for a specific role in handling the model's training, evaluation, and text generation workflow.

1. `build_parser()`:
   Creates and configures an `argparse.ArgumentParser` for command-line argument parsing. It specifies arguments for the training, evaluation, and generation of text using the model.

2. `load(args)`:
   Loads datasets for training, validation, and testing from the location specified in the arguments. It will raise errors if datasets are missing or empty when required for training or testing.

3. `loss(model, inputs, targets, lengths)`:
   Calculates the cross-entropy loss for the given inputs and targets using the provided model. It masks padded tokens, computes the loss, and normalizes it by the number of tokens (excluding padding).

4. `iterate_batches(dset, tokenizer, batch_size, train=False)`:
   Generates batches of data for training or evaluation. If `train` is `True`, the data is shuffled. It also handles token encoding, padding, and sequence length issues before yielding batches and lengths.

5. `evaluate(model, dataset, loss, tokenizer, batch_size, num_batches)`:
   Evaluates the given model on a dataset by iterating over a specified number of batches, computing the loss, and averaging it across all tokens.

6. `train(model, train_set, val_set, optimizer, loss, tokenizer, args)`:
   The main function for training the model. It runs a training loop over the iterations, computes gradients, updates the model, and periodically prints out training and validation losses.

7. `generate(model, prompt, tokenizer, args)`:
   Generates text from a given prompt using the model. It repeatedly produces one token at a time based on the model's predictions and sampling temperature specified in the arguments until the desired number of tokens is generated.

8. `load_model(folder: str)`:
   Loads a pre-trained model and its tokenizer from a specified directory. It handles loading of model configurations, quantization if specified, and setting the model weights.

### Examples and Implications of Arguments in `lora.py`

1. `--model`: ex) path/to/mistral-7B-v0.1
   - **Purpose**: Specifies the path to the model directory.
   - **Implication**: This path should contain necessary model files like tokenizer, weights, and configuration files. It's crucial for loading the pre-trained model before finetuning or generation.

2. `--num-tokens` (`-n`): ex) 200
   - **Purpose**: Defines how many tokens the model should generate.
   - **Implication**: Used in the text generation phase, controlling the length of the generated output.

3. `--write-every`: ex) 10
   - **Purpose**: Determines after how many tokens the output should be detokenized and displayed/written.
   - **Implication**: Affects the frequency of feedback during generation, useful for monitoring the ongoing generation process.

4. `--temp`: ex) 1.0
   - **Purpose**: Sets the sampling temperature for generation.
   - **Implication**: A higher temperature leads to more random outputs, while a lower temperature results in more predictable outputs.

5. `--prompt` (`-p`): "Once upon a time, there was a"
   - **Purpose**: Provides a starting prompt for text generation.
   - **Implication**: The prompt initializes the context for generation, guiding the model's output.

6. `--train`: 
   - **Purpose**: A flag to indicate whether the model should be trained.
   - **Implication**: Determines if the script should enter the training mode.

7. `--data`: ex) ./mlx-doc-data
   - **Purpose**: Specifies the directory containing training, validation, and test datasets (in JSONL format).
   - **Implication**: Essential for loading datasets necessary for training and evaluation.

8. `--lora-layers`: ex) 8
   - **Purpose**: Indicates the number of layers to finetune using LoRA.
   - **Implication**: Impacts the extent of model adaptation and computational efficiency of the finetuning process. (Fine-tunes the top 8 layers of the model using the LoRA technique.)

    If the user does not explicitly specify the number of layers for LoRA training, the default value is set to 16 layers. This feature allows users to experiment with fine-tuning different numbers of layers in their model. Users can choose to adapt a smaller or larger portion of the model based on their specific needs and the computational resources available to them. The number of layers fine-tuned using LoRA can significantly affect the model's performance. Fine-tuning more layers might lead to better performance since more of the model is being adapted to the specific task. However, this could also increase the risk of overfitting, especially with smaller datasets. 

    Fine-tuning fewer layers is computationally more efficient and faster. This is particularly beneficial when resources are limited or when rapid prototyping is required. It allows for a more resource-conscious approach to adapting large models.

9. `--batch-size`: ex) 8
   - **Purpose**: Sets the minibatch size for training.
   - **Implication**: Affects memory usage and training dynamics. Larger batches provide more stable gradient estimates but require more memory.

10. `--iters`: ex) 1000
    - **Purpose**: Defines the total number of training iterations.
    - **Implication**: Controls the duration of the training process. (1000 training iterations.)

11. `--val-batches`: ex) 50
    - **Purpose**: Specifies the number of validation batches to use.
    - **Implication**: Impacts the validation process during training. Setting `-1` uses the entire validation set. (Uses 50 batches from the validation set for performance evaluation during training.)

12. `--learning-rate`: ex) 5e-6
    - **Purpose**: Sets the learning rate for the Adam optimizer.
    - **Implication**: Crucial for training dynamics; too high can prevent convergence, too low can slow down training. (Sets a slightly lower learning rate for fine-tuning.)

13. `--steps-per-report`: ex) 20
    - **Purpose**: Determines the frequency of training loss reporting.
    - **Implication**: Provides regular feedback on training progress without overwhelming with too much information. (Reports training loss and other metrics every 20 steps.)

14. `--steps-per-eval`: ex) 100
    - **Purpose**: Sets how often to perform validation during training.
    - **Implication**: Allows periodic monitoring of model performance on the validation set.

15. `--resume-adapter-file`: ex) path/to/adapter_weights.npz
    - **Purpose**: Path to resume training with specific adapter weights.
    - **Implication**: Enables continuation of training from a previous state, useful for long training processes or refining already fine-tuned models.

16. `--adapter-file`: ex) adapters_preprocessed.npz
    - **Purpose**: Specifies the file path for saving or loading trained adapter weights.
    - **Implication**: Important for persisting the results of finetuning and using them in later sessions or analyses. (Text Dataset was preprocessed and saved as adapters_preprocessed.npz.)

17. `--test`: 
    - **Purpose**: Flag to indicate evaluation on the test dataset after training.
    - **Implication**: Activates the testing phase to assess model performance on unseen data.

18. `--test-batches`: ex) 100
    - **Purpose**: Determines the number of test set batches for evaluation.
    - **Implication**: Affects the extent of the testing process. Setting `-1` evaluates the entire test set.

19. `--seed`: ex) 42 
    - **Purpose**: Sets the seed for the Pseudo Random Number Generator (PRNG).
    - **Implication**: Ensures reproducibility of results by initializing random processes in a consistent manner.


## My Take on Creating Custom Datasets

[cwk_create_dataset.py](cwk_create_dataset.py)

The idea is straightforward: Mistral or Llama models can be fine-tuned with you own datasets using Apple MLX LoRa example. The problem is getting the dataset ready.

The official MLX documentation was crafted using Sphinx, which extracts docstrings directly from the MLX package modules.

To align with this methodology, I created a Python script that compiles docstrings from a specified list of packages and outputs them into a JSONL format, similar to existing example. For instance, the docstring for the module `mlx.nn.layers.activations.ReLU()` is transformed into:

```json
{
  "text": "Q: What is ReLU in mlx?\nA: mlx.nn.ReLU Applies the Rectified Linear Unit.\n\nSimply ``mx.maximum(x, 0)``."
}
```

❗️Training the model is generally uncomplicated, producing discernible results during inference. It's important to note, though, that simultaneous GPU usage by other applications can cause disruptions, such as unexpected 'nan' values during the training phase. This was observed when I was unable to record the training procedure due to conflicts with Screenflow. For example, if Screenflow remained active post-MLX usage (alongside the MLX Stable Diffusion WebUI), even video rendering applications like Final Cut encountered issues. The remedy often required closing all GPU-consuming applications, including my MLX SD WebUI. In worst cases, I need to  reboot. This might be MacOS or Metal bugs, but it's worth noting. During training I even encountered kernel panics twice. I suggest refraining from using the GPU for other tasks during training like screen recording or video rendering.

Nonetheless, training LoRA with personal datasets is impressively straightforward. The most challenging aspect tends to be preparing the dataset. Fortunately, since docstrings are readily available in any package, all that's needed is a script to extract them.

The included script `cwk_create_dataset.py` serves precisely this purpose, with configurations as follows:

```python
SEQ_LIMIT = 2048  # Controls the maximum length of extracted docstrings
TEST_SET_SIZE = 0.1  # Determines the proportion of data designated for the test set
PACKAGE_NAMES = ['mlx', ]  # Identifies the packages to source docstrings from
DATA_FOLDER = 'mlx-doc-data'  # Designates the directory for storing the dataset
DATASET = {'train.jsonl': None, 'valid.jsonl': None, 'test.jsonl': None}  # Structures the dataset for training, validation, and testing
BATCH_SIZE = 1000  # Specifies the number of docstrings to process concurrently
PREPROCESS = True  # Toggles the preprocessing of docstrings
```

Note that including different packages for recursive importing can lead to failures, with 'mlx' being a reliable standalone option. Furthermore, the script uses parallel processing for efficiency.

When initiating the training process, remember to set the args correctly:  `--data mlx-doc-data` `--model your_model_path`. . Or just edit and run the `run-training.sh` and `run-inference.sh` shell scripts.

Bear in mind that LoRA is designed to enhance the model's existing knowledge base, not to supplant the foundational pre-trained model. Its purpose is to fine-tune the model to better suit your particular dataset, which lies at the very core of its functionality. For example, while the model is already familiar with the concept of the 'ReLU' activation function, employing LoRA involves expanding its understanding by providing specific contextual information—like explaining 'ReLU' in the context of MLX—and an expected response, such as "In MLX, nn.ReLU applies the Rectified Linear Unit function, essentially executing `mx.maximum(x, 0)`." The extent to which the model integrates this new knowledge is ultimately determined by its underlying learning algorithms. In essence, the learning process of the model remains an enigma, highlighted by the fact that even renowned pioneers in AI have expressed significant reservations.

Just experiment with your own datasets and see what happens. You might be surprised by the results. I certainly was.

You may have your own perspective on custom datasets, which I value. I invite you to consider my insights on the subject in the accompanying essay:

[The-History-of-Human-Folly.md](..%2F..%2Fessays%2FAI%2FThe-History-of-Human-Folly.md)