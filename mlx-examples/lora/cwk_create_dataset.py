import json
import importlib
import inspect
import pkgutil
import os
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


SEQ_LIMIT = 2048  # Limit the sequence length for docstrings
TEST_SET_SIZE = 0.1  # Proportion of data to be used as the test set
PACKAGE_NAMES = ['mlx', ]  # List of packages to extract docstrings from
DATA_FOLDER = 'mlx-doc-data'  # Folder name where the data will be stored
DATASET = {'train.jsonl': None, 'valid.jsonl': None, 'test.jsonl': None}  # Dictionary to hold train, valid, and test data
BATCH_SIZE = 1000  # The number of docstrings written to file in one go
PREPROCESS = False  # Whether to preprocess the docstrings or not

if PREPROCESS:
    # Download nltk data
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Function to preprocess text
    def preprocess_text(text):
        # Lowercase the text
        text = text.lower()
        # Remove special characters
        text = re.sub(r'\W', ' ', text)
        # Remove single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # Substitute multiple spaces with single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize the words
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)


# Function to extract docstrings from modules
def extract_docstrings(module, inspected_modules=None):
    if inspected_modules is None:
        inspected_modules = set()

    my_docstrings = {}

    def is_inspectable(obj):
        return any([inspect.isfunction(obj), inspect.ismethod(obj), inspect.ismethoddescriptor(obj), inspect.isdatadescriptor(obj)])

    for name, obj in inspect.getmembers(module):
        if inspect.ismodule(obj) and obj.__name__ not in inspected_modules:
            inspected_modules.add(obj.__name__)
            try:
                my_docstrings.update(extract_docstrings(obj, inspected_modules))
            except (ModuleNotFoundError, RuntimeError):
                print(f"Error: The module '{obj.__name__}' could not be imported. Skipping.")
                continue
        elif inspect.isclass(obj) or is_inspectable(obj):
            docstring = inspect.getdoc(obj)
            if docstring:
                my_docstrings[f"{module.__name__}.{name}"] = docstring

    return my_docstrings


# Function to extract docstrings from all modules in a package
def extract_docstrings_from_package(package_name):
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        print(f"Error: The package '{package_name}' is not installed.")
        return {}

    my_docstrings = extract_docstrings(package)

    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        try:
            my_docstrings.update(extract_docstrings(importlib.import_module(name)))
        except (ImportError, RuntimeError):
            print(f"Error: The module '{name}' could not be imported. Skipping.")
            continue

    return my_docstrings


# Worker function for parallel processing
def worker(package_name):
    # Call the function to extract docstrings from a package
    return extract_docstrings_from_package(package_name)


# Main execution block
if __name__ == "__main__":
    # Create a pool of workers for parallel execution
    with Pool() as p:
        # Map the worker function to each package
        results = p.map(worker, PACKAGE_NAMES)

    # Combine the results from all the workers
    docstrings = {}
    for result in results:
        docstrings.update(result)

    # Apply preprocessing to docstrings, Shorten docstrings and create a list of tuples
    if PREPROCESS:
        docstrings_list = [(name, preprocess_text(doc[:SEQ_LIMIT])) for name, doc in docstrings.items() if doc]
    else:
        docstrings_list = [(name, doc[:SEQ_LIMIT]) for name, doc in docstrings.items() if doc]

    # Split the data into training, validation, and test sets
    train, test = train_test_split(docstrings_list, test_size=TEST_SET_SIZE, random_state=42)
    test, valid = train_test_split(test, test_size=0.5, random_state=42)

    # Assign the split data to the respective keys in the dictionary
    DATASET['train.jsonl'] = train
    DATASET['valid.jsonl'] = valid
    DATASET['test.jsonl'] = test

    # Create a data folder if it doesn't exist
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Write each dataset to a jsonl file
    for filename, dataset in DATASET.items():
        filepath = os.path.join(DATA_FOLDER, filename)
        with open(filepath, 'w') as file:
            # Process the dataset in batches
            for i in range(0, len(dataset), BATCH_SIZE):
                batch = dataset[i:i+BATCH_SIZE]
                # Write each docstring as a JSON object
                for name, doc in batch:
                    if not any(name.startswith(package_name) for package_name in PACKAGE_NAMES):
                        continue
                    question = f"What is {name.split('.')[-1]} in {name.split('.')[0]}?"
                    data_str = {"text": f"Q: {question}\nA: {name} {doc}"}
                    print(data_str)
                    file.write(json.dumps(data_str) + "\n")

    # Output the number of entries in the training dataset
    print(f"{len(train)} entries in the training dataset.")