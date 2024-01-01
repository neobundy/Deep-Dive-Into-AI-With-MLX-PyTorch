# Lora MLX Example - Demystification is in progress.

This guide delves into the core algorithms only. In crafting this documentation, I joined forces with GitHub Copilot to enhance the learning process. As I'm on a learning journey myself, please note there may be inaccuracies in my explanations 🤗.

## convert.py

The `convert.py` script is used to convert a PyTorch model to a format that can be used with MLX. 

1. It parses command-line arguments to get the directories of the PyTorch model and the destination for the converted MLX model.

👉 ex) mistral-7B-v0.1 -> mistral-7B-v0.1-mlx

2. It checks if the destination directory exists, and if not, it creates it.

3. It copies the tokenizer model from the PyTorch model directory to the MLX model directory.

👉 ex) tokenizer.model ->  tokenizer.model

4. It loads the PyTorch model's state dictionary (which includes the model's weights and biases) from a `.pth` file and saves it into a NumPy `.npz` file in the MLX model directory. The weights are converted to half-precision floating-point format (`float16`) to reduce memory usage.

5. It reads a `params.json` file from the PyTorch model directory, which contains various configuration parameters for the model. It removes certain parameters, adds others if they're not present, and modifies some based on the model's state dictionary. It then writes the updated configuration parameters to a new `params.json` file in the MLX model directory.

In step 4, the code performs two main operations related to loading and saving model weights.

1. `state = torch.load(str(torch_path / "consolidated.00.pth"))`: This line is loading a PyTorch model's state dictionary from a file named `consolidated.00.pth` located in the directory specified by `torch_path`. The state dictionary includes the model's weights and biases, stored as tensors.

2. `np.savez(str(mlx_path / "weights.npz"), **{k: v.to(torch.float16).numpy() for k, v in state.items()})`: This line is saving the model's state dictionary into a NumPy `.npz` file located in the directory specified by `mlx_path`. The state dictionary is first converted to a dictionary of NumPy arrays with half-precision floating-point format (`float16`) to reduce memory usage. The `**` operator is used to unpack the dictionary into keyword arguments, where each key-value pair represents a tensor's name and its corresponding values. The `.npz` format allows saving multiple arrays into a single file in a compressed format.

The `consolidated.00.pth` and `weights.npz` files both contain the model's weights, but they are in different formats and precisions. The `consolidated.00.pth` file is a PyTorch file that contains the model's weights in the original precision (usually `float32`).

The `weights.npz` file is a NumPy file that contains the same weights but converted to half-precision floating-point format (`float16`). This conversion is done to reduce the memory usage of the model's weights.

So, while the actual values of the weights are the same in both files, their storage format and precision are different.

In step 5, the code block reads a JSON configuration file(params.json), removes unnecessary keys, adds missing keys with default values, and writes the updated configuration back to a new file.

Refer to models.py explanation for what each parameter means in the model architecture.

## models.py


The `ModelArgs` class is a simple data class in Python that is used to store various parameters needed for the model. It uses Python's type hinting to specify the type of each parameter. Here's a brief explanation of each parameter:

- `dim`: This is the dimensionality of the model. In the context of transformers, this is usually the dimensionality of the embeddings.
- `n_layers`: This is the number of layers in the transformer model.
- `head_dim`: This is the dimensionality of each head in the multi-head attention mechanism of the transformer.
- `hidden_dim`: This is the dimensionality of the hidden layer in the feed-forward network of the transformer.
- `n_heads`: This is the number of heads in the multi-head attention mechanism of the transformer.
- `n_kv_heads`: This is the number of key/value heads in the multi-head attention mechanism of the transformer.
- `norm_eps`: This is the epsilon value used for normalization in the transformer model to avoid division by zero.
- `vocab_size`: This is the size of the vocabulary, i.e., the number of unique words in the dataset. This is used to create the word embeddings in the transformer model.

👉 Mistral's `params.json`: 

{
    "dim": 4096,
    "n_layers": 32,
    "head_dim": 128,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "sliding_window": 4096,
    "vocab_size": 32000
}

👉  Converted MLX `params.json` (the same):

{"dim": 4096, "n_layers": 32, "head_dim": 128, "hidden_dim": 14336, "n_heads": 32, "n_kv_heads": 8, "norm_eps": 1e-05, "vocab_size": 32000}

In `convert.py`, `n_kv_heads`, `head_dim`, and `hidden_dim` are created if they are not present in the original `params.json`.

The Q, K, V attention mechanism is a key component of Transformer models, used in natural language processing tasks.

- Q, K, and V stand for Query, Key, and Value respectively. These are vectors generated from the input data.

- The Query and Key vectors are used to compute an attention score. This score determines how much focus should be put on each part of the input when producing the output. The score is calculated by taking the dot product of the Query and Key, and then applying a softmax function to get probabilities.

- The Value vector is used to compute the final output. Each Value is multiplied by the corresponding attention score, and then all the results are summed up to produce the output.

In essence, the Q, K, V attention mechanism allows the model to focus on different parts of the input when producing each part of the output, which is particularly useful for tasks like machine translation where the relationship between input and output isn't one-to-one.

❗️Please note that a solid understanding of modern linear algebra concepts is essential to fully comprehend all of these.

The dot product is a measure of similarity between two vectors, akin to cosine similarity. It computes the sum of the multiplication of corresponding elements in the two vectors. When the vectors are normalized (i.e., their magnitudes are 1), the dot product equals the cosine of the angle between them, indicating how similar the vectors are in terms of direction. A dot product of 1 signifies that the vectors are identical, a dot product of 0 signifies that the vectors are orthogonal (i.e., unrelated), and a dot product of -1 signifies that the vectors are diametrically opposed.

Cosine similarity, on the other hand, measures the cosine of the angle between two vectors. A cosine similarity of 1 (or 0 degrees) indicates that the vectors are identical, a cosine similarity of 0 (or 90 degrees) indicates no relation, and a cosine similarity of -1 (or 180 degrees) indicates that the vectors are perfectly opposite.

The scaled dot product attention mechanism is a key component of the Transformer model, which was introduced in the paper "Attention is All You Need".

The formula for the scaled dot product attention is:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Here's a simple explanation of each part:

- `Q`, `K`, and `V` are matrices representing sets of queries, keys, and values, respectively. They are produced by linear transformations of the input.

- `QK^T` is the dot product of the query and key matrices. This calculates the similarity score for each query-key pair. The more similar a query is to a key, the higher the score will be.

- `sqrt(d_k)` is the square root of the dimension of the key vectors (`d_k`). This scaling factor is used to prevent the dot product results from growing too large, which could cause the softmax function to have extremely small gradients. This is particularly important when dealing with large input sequences.

- `softmax(QK^T / sqrt(d_k))` applies the softmax function to the scaled dot product scores. This normalizes the scores, so they sum to 1, and can be interpreted as probabilities.

- `softmax(QK^T / sqrt(d_k))V` is the weighted sum of the value vectors, where the weights are the softmax-normalized scores. This produces the output of the attention mechanism.

Why the transpose operations? 

Transposition is a fundamental operation in linear algebra that is used extensively in machine learning. Transposition is the operation of swapping the rows and columns of a matrix. If you have a matrix A, the transpose of A (denoted as A^T) is a new matrix where the row i in A becomes column i in A^T.

When you want to perform a dot product operation between two tensors (or matrices) with different shapes, you often need to transpose one of them to align their dimensions correctly. 

In the context of matrix multiplication, the number of columns in the first matrix must match the number of rows in the second matrix. If this condition isn't met, you can't perform the multiplication. Transposing one of the matrices can align the dimensions appropriately.

For example, if you have a matrix `A` of shape `(m, n)` and another matrix `B` of shape `(p, q)`, you can't directly compute the dot product `AB` because the inner dimensions `n` and `p` don't match. However, if you transpose `B` to get `B^T` of shape `(q, p)`, you can now compute the dot product `AB^T` because the inner dimensions `n` and `q` do match.

In the context of machine learning and specifically in the attention mechanism, transposition is used to align the dimensions of the Query and Key matrices so that the dot product can be computed, which is a crucial step in calculating the attention scores.

The transpose operation (`^T`) is used here to align the dimensions of the Query (`Q`) and Key (`K`) matrices so that the dot product can be computed. 

In Transformer models, the Query and Key matrices are typically of shape `(batch_size, num_heads, seq_length, depth)`. The dot product operation requires the inner dimensions of the two matrices to be the same. Therefore, we transpose the Key matrix to get a shape of `(batch_size, num_heads, depth, seq_length)`. Now, the dot product `QK^T` results in a matrix of shape `(batch_size, num_heads, seq_length, seq_length)`, where each entry at position `(i, j)` represents the attention score between the `i`-th query and the `j`-th key.

In essence, the scaled dot product attention mechanism allows the model to focus on different parts of the input when producing each part of the output, which is particularly useful for tasks like machine translation where the relationship between input and output isn't one-to-one.

To be continued...