import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
from menny_dataset import generate_dataset

NUM_SAMPLES = 500
NUM_EPOCHS = 10
QUERY_LENGTH = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_NAME = 'Menny'

# Convert characters to integers
CHAR_TO_INT = {char: i for i, char in enumerate(string.ascii_letters)}
INT_TO_CHAR = {i: char for char, i in CHAR_TO_INT.items()}


# Encoding and decoding functions
def encode_string(s):
    return [CHAR_TO_INT[char] for char in s]


def decode_array(arr):
    return [INT_TO_CHAR.get(i, '?') for i in arr]


def prepare_tensors(strings, max_length, input_size):
    tensors = []
    for string in strings:
        tensor = np.zeros((max_length, input_size))
        for i, char_idx in enumerate(string):
            tensor[i][char_idx] = 1
        tensors.append(torch.tensor(tensor, dtype=torch.float32))
    return tensors


class Menny(nn.Module):
    def __init__(self):
        super(Menny, self).__init__()
        num_heads = 4
        self.transformer = nn.Transformer(input_size, num_heads)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        x = self.transformer(x, x)
        x = x.contiguous().view(x.size(0), -1, x.size(-1))  # Use the size of the first and last dimensions of x
        x = self.fc(x)
        return x


def train(model, batch_size):
    assert isinstance(input_tensors, list), "input_tensors should be a list of tensors"
    assert isinstance(label_tensors, list), "label_tensors should be a list of tensors"

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create DataLoader for batching
    dataset = list(zip(input_tensors, label_tensors))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    for epoch in range(NUM_EPOCHS):
        print(f'Starting epoch {epoch + 1}/{NUM_EPOCHS}')  # Print the start of each epoch
        for i, (input_batch, target_batch) in enumerate(data_loader):
            # print(f'Starting iteration {i + 1} in epoch {epoch + 1}')  # Print the start of each iteration
            optimizer.zero_grad()
            output = model(input_batch)

            # Ensure the output tensor and the target tensor have the same shape
            output = output.view(target_batch.shape)
            target_batch = target_batch.view(output.shape)

            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 9:
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss.item()}')


# Testing
def test_model(model, test_data, test_labels):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in zip(test_data, test_labels):
            outputs = model(data)
            outputs = outputs.view(label.shape)  # Ensure the output tensor and the label tensor have the same shape
            predicted = torch.argmax(outputs, dim=1)
            total += label.size(0)
            correct += (predicted == torch.argmax(label, dim=1)).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def query_model(model):
    # Ask for user input
    input_string = input(f"Enter your prompt(max {QUERY_LENGTH} chars,  q to quit): ")
    if input_string == 'q':
        return None

    # Truncate the input_string to match QUERY_LENGTH
    input_string = input_string[:QUERY_LENGTH]

    # Ensure the model is in evaluation mode
    model.eval()

    # Encode the input string
    encoded_input = encode_string(input_string)

    # Prepare the tensor
    input_tensor = prepare_tensors([encoded_input], max_length, input_size)

    # Get the model's output
    with torch.no_grad():
        output = model(input_tensor[0])

    # Ensure the output is within the range of the dictionary keys
    output = torch.clamp(output, 0, len(INT_TO_CHAR) - 1)

    # Reshape the output tensor to match the shape of the input tensor
    output = output.view(input_tensor[0].shape)  # Use the shape of the first tensor in the list

    # Decode the output to a string
    output_indices = torch.argmax(output, dim=1)
    response = decode_array(output_indices.numpy().flatten())  # Flatten the numpy array

    # Join the characters in the list into a single string
    response = ''.join(response)

    # Truncate the response to match the length of the input string
    response = response[:len(input_string)]

    return response


if __name__ == '__main__':
    # Prepare the dataset
    query_list, label_list = generate_dataset(NUM_SAMPLES)
    encoded_queries = [encode_string(query) for query in query_list]
    encoded_labels = [encode_string(label) for label in label_list]

    input_size = len(string.ascii_letters)
    max_length = QUERY_LENGTH

    # Convert lists to PyTorch tensors
    input_tensors = prepare_tensors(encoded_queries, max_length, input_size)
    label_tensors = prepare_tensors(encoded_labels, max_length, input_size)

    model = Menny()

    # Train Menny
    train(model, BATCH_SIZE)

    accuracy = test_model(model, input_tensors, label_tensors)
    print(f'Accuracy of {MODEL_NAME} on the test set: {accuracy:.2f}%')

    while True:
        response = query_model(model)
        if response is None:
            break
        print(f'{MODEL_NAME}: {response}')


