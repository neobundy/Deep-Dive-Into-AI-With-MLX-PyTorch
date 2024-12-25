import json
import mlx.data as dx

DATA_FOLDER = './data'

def load_jsonl_in_chunks(file_path, chunk_size=1000):
    """ Generator that yields chunks of data from a JSONL file. """
    chunk = []
    with open(file_path, 'r') as file:
        for line in file:
            value = json.loads(line)['text']
            value = value.encode('utf-8')  # Byte encode the value
            chunk.append({'text': value}) # Add the byte-encoded value to the chunk
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:  # Yield the last chunk if it's not empty
            yield chunk

def create_stream_from_jsonl(file_path, chunk_size=1000):
    """ Creates a stream from a JSONL file processed in chunks. """
    for chunk in load_jsonl_in_chunks(file_path, chunk_size):
        buffer = dx.buffer_from_vector(chunk)
        stream = buffer.to_stream()
        yield from stream.prefetch(prefetch_size=10, num_threads=4)

# Creating streams for each dataset
train_stream = create_stream_from_jsonl(DATA_FOLDER + '/train.jsonl')
valid_stream = create_stream_from_jsonl(DATA_FOLDER + '/valid.jsonl')
test_stream = create_stream_from_jsonl(DATA_FOLDER + '/test.jsonl')

# Print the first sample from each dataset: byte encoded but not decoded
print("First Train Sample:", next(iter(train_stream))['text'])
print("First Valid Sample:", next(iter(valid_stream))['text'])
print("First Test Sample:", next(iter(test_stream))['text'])

# Print the first sample from each dataset: byte encoded and decoded
byte_string = bytes(next(iter(train_stream))['text'])
print(f"First Train Sample Decoded: {byte_string.decode('utf-8')}")

