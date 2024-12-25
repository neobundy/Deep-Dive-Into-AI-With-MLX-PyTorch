import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.data as dx
from transformers import AutoTokenizer
import json
import time

# Define the data directory and model parameters
# Demonstration purposes only, do not use this model for production

DATA_FOLDER = './data'
INPUT_SIZE = 768
NUM_EPOCHS = 10
CHUNK_SIZE = 100
BATCH_SIZE = 32


def load_jsonl_in_chunks(file_path, chunk_size=CHUNK_SIZE):
    """
    Generator that yields chunks of data from a JSONL file.
    Each chunk contains up to `chunk_size` lines from the file.
    """
    chunk = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a JSON object and encode the text value as bytes
            value = json.loads(line)['text']
            value = value.encode('utf-8')
            chunk.append({'text': value})

            # When the chunk reaches the specified size, yield it and start a new chunk
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []

        # Yield any remaining data as the last chunk
        if chunk:
            yield chunk


def create_stream_from_jsonl(file_path, chunk_size=CHUNK_SIZE):
    """
    Creates a stream of data from a JSONL file, processed in chunks.
    Each yielded item is a batch of data pre-processed for MLX data loading.
    """
    for chunk in load_jsonl_in_chunks(file_path, chunk_size):
        buffer = dx.buffer_from_vector(chunk)
        stream = buffer.to_stream()
        yield from stream.prefetch(prefetch_size=10, num_threads=4)


def prepare_and_tokenize_data(stream, tokenizer, max_length, batch_size=BATCH_SIZE):
    """
    Prepares and tokenizes data from a given stream into batches.
    Each batch contains tokenized data of size `batch_size`.
    """
    for chunk in stream:
        # Decode each chunk into a string
        decoded_text = bytes(chunk['text']).decode('utf-8')

        # Tokenize the decoded text
        tokenized_inputs = tokenizer(decoded_text, padding='max_length', truncation=True, max_length=max_length,
                                     return_tensors="np")

        # Yield batches of tokenized data
        total_samples = tokenized_inputs['input_ids'].shape[0]
        for start_idx in range(0, total_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_input_ids = tokenized_inputs['input_ids'][start_idx:end_idx]
            batch_attention_mask = tokenized_inputs['attention_mask'][start_idx:end_idx]
            yield batch_input_ids, batch_attention_mask


def loss_fn(model, input_ids, attention_mask):
    """
    Computes the loss function for the given model and data.
    The function returns both the loss and the perplexity.
    """
    # Forward pass through the model
    output = model(mx.array(input_ids), mx.array(attention_mask))

    # Prepare target data by shifting input_ids
    targets = mx.roll(mx.array(input_ids), shift=-1, axis=1)

    # Flatten the output and targets for cross-entropy loss calculation
    output_flat = output.reshape(-1, output.shape[-1])
    targets_flat = targets.reshape(-1)

    # Calculate the loss and perplexity
    loss = nn.losses.cross_entropy(output_flat, targets_flat, reduction='none').mean()
    perplexity = mx.exp(loss)
    return loss, perplexity


# Our trusty Menny Transformer Model, but again, she's just for demonstration purposes only. She doensn't actually work.
class Menny(nn.Module):
    def __init__(self):
        super(Menny, self).__init__()
        num_heads = 4
        # Define a simple transformer model
        self.transformer = nn.Transformer(INPUT_SIZE, num_heads)
        self.fc = nn.Linear(INPUT_SIZE, INPUT_SIZE)

    def __call__(self, x):
        # Forward pass through the transformer and a linear layer
        x = self.transformer(x, x, None, None, None)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.fc(x)
        return x.reshape(-1, INPUT_SIZE)


# Initialize the model, tokenizer, and optimizer
menny = Menny()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_length = 128  # Define the maximum token sequence length
optimizer = optim.SGD(learning_rate=0.01)

# Training loop
i = 0
for epoch in range(NUM_EPOCHS):
    # Recreate the train_stream at the start of each epoch
    train_stream = create_stream_from_jsonl(DATA_FOLDER + '/train.jsonl')

    # Prepare and tokenize data
    train_data = prepare_and_tokenize_data(train_stream, tokenizer, max_length)

    print(f"Simulating training step with input_ids and attention_mask for epoch {epoch + 1}:")
    time.sleep(0.5)
    for input_ids, attention_mask in train_data:
        print("Input IDs:", input_ids.shape)
        print("Attention Mask:", attention_mask.shape)
        # Additional training steps (loss computation, model updates, etc.)

