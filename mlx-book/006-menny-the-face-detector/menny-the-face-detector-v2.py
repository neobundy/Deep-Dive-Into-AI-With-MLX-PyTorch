import mlx.core as mx
import mlx.nn as nn
import mlx.data as dx
import mlx.optimizers as optim
from pathlib import Path
import streamlit as st
import os
from PIL import Image
import numpy as np

# Title for Streamlit App
TITLE = "Menny, the Face Detector"

# Define the base directory
IMAGE_FOLDER = './portraits'
IMAGE_TYPE = ".png"

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# Conversion dictionary from numerical labels to corresponding person names
LABEL_TO_NAME = {0: "Cody", 1: "CWK", 2: "Pippa", 3: "RJ"}

# HYPERPARAMETERS

FORCE_TRAINING = False
USE_COMPLEX_MODEL = True

NUM_EPOCHS = 100
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
LEARNING_RATE = 0.00001
FLIP_PROBABILITY = 0.5
ROTATION_ANGLE = 15

MODEL_WEIGHTS_FILE = 'menny_model_weights.npz'

def portraits_and_labels(image_folder: Path):
    """
    Load the files and classes from an image dataset that contains one folder per class.
    Each subfolder under 'image_folder' is a category, and contains image files.
    """
    categories = [f.name for f in image_folder.iterdir() if f.is_dir()]
    category_map = {c: i for i, c in enumerate(categories)}

    images = []
    for category in categories:
        category_path = image_folder / category
        # Print all files in the category directory for debugging
        all_files = list(category_path.iterdir())

        # Check for image files specifically, regardless of case
        category_images = [img for img in category_path.iterdir() if img.suffix.lower() == IMAGE_TYPE]
        images.extend(category_images)

    print(f"Found {len(images)} images in {len(categories)} categories.")
    return [
        {
            "image": str(p).encode("ascii"),  # Use full path
            "label": category_map[p.parent.name]
        }
        for p in images
    ], len(categories)


class SimpleFaceDetector(nn.Module):
    def __init__(self):
        super(SimpleFaceDetector, self).__init__()

        # Reduced number of convolutional layers and filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Simplified fully connected layers
        self.fc1 = nn.Linear(input_dims=32 * 128 * 128, output_dims=128)
        self.fc2 = nn.Linear(input_dims=128, output_dims=4)  # Assuming 4 classes

        # Other layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def __call__(self, x):
        # Ensure input tensor x is of correct type and shape
        if not isinstance(x, mx.array):
            # Convolution currently only supports floating point types
            x = mx.array(x, dtype=mx.float32)
        if len(x.shape) != 4:
            raise ValueError("Input tensor must have 4 dimensions (NHWC format)")

        # Convolutional layers with pooling and activations
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        x = pool(self.relu(self.conv1(x)))
        x = pool(self.relu(self.conv2(x)))

        # Flattening and fully connected layers
        x = mx.flatten(x, start_axis=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        x = mx.softmax(x, axis=1)

        return x


class ComplexFaceDetector(nn.Module):
    def __init__(self):
        super(ComplexFaceDetector, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Dropout layer
        self.dropout = nn.Dropout(DROPOUT_RATE)

        # Fully connected layers
        self.fc1 = nn.Linear(input_dims=128 * 64 * 64, output_dims=512)
        self.fc2 = nn.Linear(input_dims=512, output_dims=4)  # Assuming 4 classes

        # Activation functions
        self.relu = nn.ReLU()

    def __call__(self, x):
        # Ensure input tensor x is of correct type and shape
        if not isinstance(x, mx.array):
            # Convolution currently only supports floating point types
            x = mx.array(x, dtype=mx.float32)
        if len(x.shape) != 4:
            raise ValueError("Input tensor must have 4 dimensions (NHWC format)")

        # Forward pass through convolutional layers with ReLU activations
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        x = pool(self.relu(self.conv1(x)))
        x = pool(self.relu(self.conv2(x)))
        x = pool(self.relu(self.conv3(x)))
        x = self.dropout(x)

        # Flattening the output for the fully connected layer
        x = mx.flatten(x, start_axis=1)  # Flatten all but the batch dimension

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)
        x = mx.softmax(x, axis=1)

        return x


def loss_fn(model, images, labels):
    # Check shapes of logits and labels
    logits = model(images)
    labels = mx.array(labels, dtype=mx.int32)
    # print(f"Labels: {labels}")
    # print(f"Types of Labels: {type(labels)}")
    # print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
    loss = nn.losses.cross_entropy(logits, labels)
    # print(f"Loss shape: {loss.shape}")  # Debugging statement

    return mx.mean(loss)


def train(model, dataset):
    # Initialize variables for Early Stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    # Composable function for loss and gradient computation
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Define the optimizer
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        stream = create_stream_from_dataset(dset)
        epoch_loss = 0.0
        batch_count = 0

        for batch in stream:
            # Extract images and labels from the batch
            images, labels = batch["image"], batch["label"]
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, images, labels)

            # Update model parameters
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            batch_count += 1

            average_epoch_loss = epoch_loss / batch_count
            print(f"Epoch {epoch}, Average Loss: {average_epoch_loss}")

            # Check for improvement
            if average_epoch_loss < best_loss:
                best_loss = average_epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping check
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {epoch + 1} epochs!')
                break

        # Save the model if early stopping didn't trigger
        if epochs_no_improve < patience:
            model.save_weights(MODEL_WEIGHTS_FILE)
            print("Model weights saved.")


def normalize(x):
    return x.astype(np.float32) / 255.0


# Function to create a data stream from the dataset
def create_stream_from_dataset(dataset):
    return (
        dataset
        .load_image("image")
        .squeeze("image")
        .image_random_h_flip("image", prob=FLIP_PROBABILITY)
        .image_rotate("image", angle=ROTATION_ANGLE)
        .image_resize("image", w=IMAGE_WIDTH, h=IMAGE_HEIGHT)
        .key_transform("image", normalize)
        .shuffle()
        .to_stream()
        .batch(BATCH_SIZE)
        .prefetch(8, 4)
    )


if __name__ == '__main__':
    if USE_COMPLEX_MODEL:
        print("Using complex model...")
        Menny = ComplexFaceDetector()
    else:
        print("Using simple model...")
        Menny = SimpleFaceDetector()

    # Check if the model weights file exists
    if os.path.exists(MODEL_WEIGHTS_FILE) and not FORCE_TRAINING:
        # Load the weights if the file exists
        print("Loading model weights...")
        Menny.load_weights(MODEL_WEIGHTS_FILE)
    else:
        # Train the model if the weights file does not exist
        print("Training model...")
        dataset, num_labels = portraits_and_labels(Path(IMAGE_FOLDER))
        dset = dx.buffer_from_vector(dataset)
        train(Menny, dset)

        # Save the trained model weights
        Menny.save_weights(MODEL_WEIGHTS_FILE)
        print("Model weights saved.")

    st.title(TITLE)

    # File upload widget
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded file and convert it to a PIL Image
        test_image = Image.open(uploaded_file)

        # Resize the image to match the input size expected by the model (512x512)
        resized_image = test_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        # Convert PIL Image to numpy array
        image_array = normalize(np.array(resized_image))

        # Add a batch dimension to the image
        image_array = np.expand_dims(image_array, axis=0)

        # Process the image through the model
        test_output = Menny(image_array)
        print(f"Test Output: {test_output}")
        predicted_label = mx.argmax(test_output, axis=1)
        print(f"Predicted Label: {predicted_label.item()}")
        # Show the prediction
        st.write(f"Predicted Label: {LABEL_TO_NAME[predicted_label.item()]}")
        # Display the uploaded image
        st.image(test_image, caption='Uploaded Image', use_column_width=True, width=IMAGE_WIDTH)



