import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import string
from menny_dataset import generate_dataset

import os

MODEL_WEIGHTS_FILE = 'menny_model_weights.npz'

NUM_SAMPLES = 500
NUM_EPOCHS = 100
QUERY_LENGTH = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_NAME = 'Menny'

# Convert characters to integers
CHAR_TO_INT = {char: i for i, char in enumerate(string.ascii_letters)}
INT_TO_CHAR = {i: char for char, i in CHAR_TO_INT.items()}

# Global lookup tables
ENCODE_LOOKUP = {char: i for i, char in enumerate(string.ascii_letters)}
DECODE_LOOKUP = {i: char for i, char in enumerate(string.ascii_letters)}


def encode_string(s):
    return [ENCODE_LOOKUP[char] for char in s]


def decode_array(arr):
    return [DECODE_LOOKUP.get(i, '?') for i in arr]


def prepare_arrays(encoded_strings, max_length, input_size, is_label=False):
    # Initialize a 3D numpy array with zeros
    arr = np.zeros((len(encoded_strings), max_length, input_size), dtype=np.float32 if not is_label else np.int32)

    # Loop through each encoded string and set the corresponding positions to 1
    for i, encoded_string in enumerate(encoded_strings):
        for j, char_idx in enumerate(encoded_string):
            if j < max_length:
                arr[i, j, char_idx] = 1

    return mx.array(arr)


class Menny(nn.Module):
    def __init__(self):
        super(Menny, self).__init__()
        num_heads = 4
        self.transformer = nn.Transformer(input_size, num_heads)
        self.fc = nn.Linear(input_size, input_size)

    def __call__(self, x):
        x = self.transformer(x, x, None, None, None)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.fc(x)
        return x.reshape(-1, input_size)  # Reshaping to [batch_size * QUERY_LENGTH, num_classes]


def loss_fn(model, input_batch, target_batch):
    output = model(input_batch)
    target_indices = mx.argmax(target_batch, axis=2)  # Convert one-hot to indices
    return nn.losses.cross_entropy(output, target_indices.reshape(-1), reduction='none').mean()


def batch_iterate(batch_size, X, y):
    num_batches = len(X) // batch_size
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        yield X[start_index:end_index], y[start_index:end_index]


def train(model, input_arrays, label_arrays):
    for epoch in range(NUM_EPOCHS):
        print(f'Starting epoch {epoch + 1}/{NUM_EPOCHS}')

        # Create batches
        for input_batch, target_batch in batch_iterate(BATCH_SIZE, input_arrays, label_arrays):
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, input_batch, target_batch)

            # Update the model
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        if epoch % 10 == 9:
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss.item()}')


def test_model(model, test_data, test_labels):
    model.eval()
    correct = 0
    total = 0

    for data, label in zip(test_data, test_labels):
        # Ensure data has batch dimension
        if len(data.shape) == 2:
            data = mx.expand_dims(data, axis=0)

        outputs = model(data)

        # Flatten the output and labels for comparison
        outputs = outputs.reshape(-1, input_size)
        label = label.reshape(-1, input_size)

        # Convert outputs to class predictions
        predicted = mx.argmax(outputs, axis=1)
        true_labels = mx.argmax(label, axis=1)

        # Update total and correct counts
        total += true_labels.size
        correct += (predicted == true_labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def query_model(model):
    # Ask for user input
    input_string = input(f"Enter your prompt (max {QUERY_LENGTH} chars, q to quit): ")
    if input_string == 'q':
        return None

    # Truncate the input_string to match QUERY_LENGTH
    input_string = input_string[:QUERY_LENGTH]

    # Ensure the model is in evaluation mode
    model.eval()

    # Encode the input string
    encoded_input = encode_string(input_string)

    # Prepare the tensor
    input_arr = prepare_arrays([encoded_input], QUERY_LENGTH, input_size)

    # Use numpy to add a batch dimension and convert back to MLX array
    input_np = np.expand_dims(np.array(input_arr[0]), axis=0)
    input_arr_batched = mx.array(input_np)

    output = model(input_arr_batched)

    # Flatten the output and decode each character
    output = output.reshape(-1, input_size)  # Flatten the output
    output_indices = mx.argmax(output, axis=1)
    response = decode_array(np.asarray(output_indices).flatten())

    # Join the characters in the list into a single string
    response = ''.join(response)

    # Truncate the response to match the length of the input string
    response = response[:len(input_string)]

    return response


if __name__ == '__main__':
    input_size = len(string.ascii_letters)
    max_length = QUERY_LENGTH

    model = Menny()

    if os.path.exists(MODEL_WEIGHTS_FILE):
        # Load previously saved weights
        model.load_weights(MODEL_WEIGHTS_FILE)
        print("Loaded model weights from saved file.")
    else:
        # Training the model if no saved weights are found
        query_list, label_list = generate_dataset(NUM_SAMPLES)
        encoded_queries = [encode_string(query) for query in query_list]
        encoded_labels = [encode_string(label) for label in label_list]

        input_arrays = prepare_arrays(encoded_queries, max_length, input_size)
        label_arrays = prepare_arrays(encoded_labels, max_length, input_size, is_label=True)

        # Function to compute both loss and gradient
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        optimizer = optim.Adam(learning_rate=LEARNING_RATE)

        train(model, input_arrays, label_arrays)

        # Saving the model weights after training
        model.save_weights(MODEL_WEIGHTS_FILE)
        print(f"Saved model weights to '{MODEL_WEIGHTS_FILE}'.")

        # Evaluate the model if it was loaded or newly trained
        accuracy = test_model(model, input_arrays, label_arrays)
        print(f'Accuracy of {MODEL_NAME} on the test set: {accuracy:.2f}%')

    # Interactive query session
    while True:
        response = query_model(model)
        if response is None:
            break
        print(f'{MODEL_NAME}: {response}')
