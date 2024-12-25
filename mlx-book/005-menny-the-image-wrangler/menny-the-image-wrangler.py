import streamlit as st
from pathlib import Path
import mlx.data as dx
import os

# Title for Streamlit App

TITLE = "Menny, the Image Wrangler"

# Define the base directory
IMAGE_FOLDER = './portraits'
IMAGE_TYPE = ".png"
# Conversion dictionary from numerical labels to corresponding person names
LABEL_TO_NAME = {0: "Cody", 1: "CWK", 2: "Pippa", 3: "RJ"}


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



# Load dataset into MLX Data buffer and get the number of unique labels
dataset, num_labels = portraits_and_labels(Path(IMAGE_FOLDER))
dset = (
    dx.buffer_from_vector(dataset)
)

# Streamlit App
st.title(TITLE)

# Display the first portrait of each person
displayed_labels = set()

# Button for random sampling
if st.button('Random Sample Portraits'):
    # Reset the stream or shuffle it to get random samples
    stream = (
        dset
        .load_image("image")
        .image_random_h_flip("image", prob=0.5)
        .shuffle()
        .to_stream()
        .batch(1)
        .prefetch(8, 4)
    )

    # Display a 4x1 grid of portraits, for 5 rows
    for _ in range(5):  # 5 rows
        cols = st.columns(4)  # 4 columns
        for col in cols:
            try:
                batch = next(stream)
                label = batch['label'][0]
                person_name = LABEL_TO_NAME.get(label, "Unknown")
                col.image(batch['image'][0], width=128, caption=f"Portrait of {person_name}")
            except StopIteration:
                break
