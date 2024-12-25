import mlx.data as dx
import json

DATA_FOLDER = './data'

# Function to load data from a jsonl file
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Load the datasets
train_data = load_jsonl(DATA_FOLDER + '/train.jsonl')
valid_data = load_jsonl(DATA_FOLDER + '/valid.jsonl')
test_data = load_jsonl(DATA_FOLDER + '/test.jsonl')


# Function to convert data into a format compatible with MLX Data
def convert_to_mlx_format(data):
    return [{'text': item['text']} for item in data]

# Convert the datasets
mlx_train_data = convert_to_mlx_format(train_data)
mlx_valid_data = convert_to_mlx_format(valid_data)
mlx_test_data = convert_to_mlx_format(test_data)

# Create MLX Data buffers from the datasets
train_buffer = dx.buffer_from_vector(mlx_train_data)
valid_buffer = dx.buffer_from_vector(mlx_valid_data)
test_buffer = dx.buffer_from_vector(mlx_test_data)

# Create a simple pipeline for each dataset
train_stream = train_buffer.to_stream()
valid_stream = valid_buffer.to_stream()
test_stream = test_buffer.to_stream()

# Print the first sample from each dataset
print("First Train Sample:", next(iter(train_stream))['text'])
print("First Valid Sample:", next(iter(valid_stream))['text'])
print("First Test Sample:", next(iter(test_stream))['text'])
