import json
import re
import random

def split_data(input_file_path, output_dir, test_ratio, val_ratio):
    with open(input_file_path, 'r') as file:
        data = file.read()

    # Define regex patterns for context and response
    context_pattern = re.compile(r'#context\n"([^"]+)"')
    response_pattern = re.compile(r'#response\n"([^"]+)"')

    # Find all matches for context and response
    contexts = context_pattern.findall(data)
    responses = response_pattern.findall(data)

    # Check if the number of contexts and responses matches
    if len(contexts) != len(responses):
        raise ValueError("The number of contexts and responses does not match.")

    # Pair contexts and responses and shuffle the dataset
    dataset = [{"text": f"#context\n\n{c}\n\n#response\n\n{r}"} for c, r in zip(contexts, responses)]
    random.shuffle(dataset)

    # Calculate dataset sizes for test, validation, and training sets
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size

    # Split the dataset
    test_set = dataset[:test_size]
    val_set = dataset[test_size:test_size + val_size]
    train_set = dataset[test_size + val_size:]

    # Save datasets in JSONL format
    for set_name, set_data in zip(["test", "valid", "train"], [test_set, val_set, train_set]):
        with open(f"{output_dir}/{set_name}.jsonl", 'w') as file:
            for item in set_data:
                json.dump(item, file)
                file.write('\n')

    return test_size, val_size, train_size

# Settings
input_file_path = 'refined-custom-dataset.md'
output_dir = './data'
TEST_RATIO = 0.1  # 10% of data for testing
VAL_RATIO = 0.1   # 10% of data for validation

test_size, val_size, train_size = split_data(input_file_path, output_dir, TEST_RATIO, VAL_RATIO)
print(f"Test set size: {test_size}, Validation set size: {val_size}, Training set size: {train_size} - Total: {test_size + val_size + train_size}")
