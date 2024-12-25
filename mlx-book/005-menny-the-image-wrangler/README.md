# Chapter 5 - Menny, the Image Wrangler

![cwk-family.jpeg](..%2F..%2Fimages%2Fcwk-family.jpeg)

In our previous exploration, we delved deeply into MLX Data using a unique, custom synthetic cynicism dataset. Now, our journey progresses into the vibrant realm of image data.

Grasping the workflow of MLX Data with images isn't overly complex, especially if you're already familiar with how MLX Data handles datasets in general. Thankfully, there are no unexpected surprises like those we encountered with our custom textual dataset.

A key point to remember is that MLX Data internally encodes image filenames as numerical values. However, this doesn't typically concern us as users. The MLX Data framework communicates internally using this encoding scheme, and unless there's a specific need to extract filenames, the encoding and decoding processes can remain largely opaque.

## CWK Family Portraits Dataset
![screenshot.png](images%2Fscreenshot.png)
Let's introduce our dataset. As with our previous data, these images are synthetic, crafted using Stable Diffusion models. They feature portraits of Cody (my AI son), Pippa (my AI daughter), RJ (my AI companion for art works), and CWK (my AI persona). Each portrait has been generated in various angles for LoRA training, offering a rich set of data.

![cody-default.png](portraits%2FCody%2Fcody-default.png)

_Cody, My AI Son_

![pippa-default.png](portraits%2FPippa%2Fpippa-default.png)

_Pippa, My AI Daughter_

![rj-blonde-front-view.png](portraits%2FRJ%2Frj-blonde-front-view.png)

_RJ, My AI Companion for CWK Art Works_

![cwk-default.png](portraits%2FCWK%2Fcwk-default.png)

_CWK, Yours Truly (Unbelievable, isn't it?)_

All images are high-resolution 512x512 pixels in RGB format, ideal for effective training.

![folders.png](images%2Ffolders.png)

The dataset is organized with a root folder named `./portraits`, where each subfolder corresponds to a family member and contains all their portraits. The images are stored in PNG format. In our dataset, each person is associated with a label. These labels are numerically encoded for processing efficiency:

```python
# Conversion dictionary from numerical labels to corresponding person names
LABEL_TO_NAME = {0: "Cody", 1: "CWK", 2: "Pippa", 3: "RJ"}
```

This dictionary maps the numerical labels to their respective human-readable names, simplifying the task of identifying each individual in the dataset.

Beware of a common misstep when working with MLX Data for images. You might encounter an I/O error indicating a JPEG loader failure:

```bash
RuntimeError: load_jpeg: could not load <CWK/left-three-quarter-view-chin-up.png>
libc++abi: terminating due to uncaught exception of type std::runtime_error: load_jpeg: could not load <RJ/rj-blonde-three-quarter-view-chin-down-4.png>
```

This error isn't indicative of MLX Data's format limitations but rather signifies general I/O issues, such as missing files or improperly constructed relative paths.

Our ultimate goal is to develop `Menny, the Face Detector`. We'll train her on this dataset to recognize and identify the faces of the CWK family's Fantastic Four. But before Menny can embark on face detection, she must first master the art of loading and manipulating image data with MLX Data. Thus begins the tale of `Menny, the Image Wrangler`.

## Handling Images with MLX Data: Buffers, Streams, and Samples

When navigating the world of MLX Data, a fundamental understanding of how it processes various data types - text, audio, images, or videos - is crucial. The essence lies in loading data into buffers and then channeling these through streams, which essentially function as generators for iteration. This core concept underpins the MLX Data architecture.

### The Cornerstones of MLX Data: Buffers, Streams, and Samples

1. **Samples:** In MLX Data, a sample is the basic unit of data. It's a dictionary that maps string keys to array values. In Python, it could be any dictionary where values adhere to the buffer protocol, while in C++, it's an instance of `std::unordered_map<std::string, std::shared_ptr<mlx::data::Array>>`. A sample can vary from a simple numeric array to a complex structured data type, such as an image file path or a textual string.

In our current example, each sample comprises the image data, which is indicated by either a relative or an absolute filepath, along with a numerical label that identifies it. Images can be loaded on a demand basis, adhering to a lazy loading approach.

2. **Buffers:** Think of a buffer as a container of samples. It's indexable, meaning you can access its elements via indices, and it has a fixed length. Importantly, buffers can be shuffled or accessed randomly. A notable feature is their ability to define operations on their samples, leading to the creation of new buffers in a lazily evaluated manner. For example, if you have a buffer containing image file paths, invoking `Buffer.load_image()` on it creates a new buffer that loads these images when needed, rather than pre-loading them into memory.

MLX Data provides a seamless avenue for augmenting your samples, a crucial feature especially when working with limited datasets. In our case, with a modest collection of portraits, we can effectively expand the dataset by applying horizontal flips. This simple transformation has the potential to double our dataset size.

Consider the following Python code snippet demonstrating this concept:

```python
stream = (
    dset
    .load_image("image")
    .image_random_h_flip("image", prob=0.5)  # 50% chance of horizontal flip
    .shuffle()
    .to_stream()
    .batch(1)
    .prefetch(8, 4)
)
```

The `image_random_h_flip("image", prob=0.5)` function within this stream pipeline randomly flips the images horizontally with a probability of 50%. This approach not only enriches the dataset but also introduces a degree of variability that can be beneficial for training robust machine learning models.

For further augmentation techniques and options, refer to the official MLX Data documentation:

https://ml-explore.github.io/mlx-data/build/html/index.html

3. **Streams:** When dealing with large or complex datasets that are either too bulky or structured in a way that inhibits random access, streams come into play. Streams in MLX Data are essentially iterables of samples with potential infinity. Unlike buffers, streams can nest within each other. This feature is especially useful when working with nested data, like reading lines from CSV files represented in a stream. Operations defined on stream samples are executed only upon accessing the sample, thereby optimizing memory and computational resources.

### Practical Application: Image Handling in MLX Data

To illustrate the application of these concepts in image handling, let's consider a dataset of family portraits, categorized by member names and stored in respective subfolders. Here's how we would load and process this image dataset using MLX Data:

```python
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


# Transforming the buffer into a stream for processing
stream = (
    dset
    .shuffle()
    .to_stream()
    .batch(32)
    .prefetch(8, 4)
)

# Iterating over the dataset
sample = next(stream)
```

This workflow exemplifies the efficiency of MLX Data in handling image datasets. Initially, the dataset is encapsulated into a buffer, setting the stage for data management. Subsequently, this buffer is transformed into a stream, paving the way for additional processing steps such as shuffling, augmenting (like horizontal flipping of images), batching, and prefetching. This meticulously crafted stream is then primed for iterative access, serving as a versatile tool for tasks ranging from training machine learning models to various other data processing activities.

## Portrait Gallery WebUI - Menny, the Image Wrangler in Action

Welcome to the unveiling of _**Menny, the Image Wrangler**_! While she may not yet be the full-fledged AI embodiment we envision, she's taking her first steps as an interactive Streamlit WebUI. Let's temper our expectations slightly – she's in her early stages, after all. And do bear with me; crafting three chapters in a single day has been a whirlwind of creativity and code! 🤣

```python
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

```

The provided code demonstrates a comprehensive application of MLX Data in handling and displaying an image dataset within a Streamlit app. Let's break down the key components and highlight important considerations, especially those learned from earlier pitfalls:

### Setup and Data Preparation

1. **Streamlit App Title**: Every great application starts with a fitting title, and ours is no exception. We're crowning our Streamlit app with the engaging moniker `Menny, the Image Wrangler`. A good title sets the stage, after all!
   
2. **Directory and Image Type Definitions**:
   - `IMAGE_FOLDER` specifies the root directory containing the image dataset.
   - `IMAGE_TYPE` is set to ".png", indicating that the images are in PNG format. 

3. **Label Mapping**: `LABEL_TO_NAME` is a dictionary that maps numerical labels to human-readable names, facilitating user-friendly display.

### Function `portraits_and_labels`

This function is responsible for loading the image dataset:
- **Directory Traversal**: It iterates through subdirectories (each representing a category) in `IMAGE_FOLDER`.
- **Image Collection**: For each category, the function gathers all images matching `IMAGE_TYPE`.
- **Sample Creation**: Each image path is paired with its corresponding label to create a sample.
- **Full Paths Encoding**: Image paths are encoded as ASCII bytes. This approach ensures compatibility with MLX Data but requires careful handling, as paths must be correctly interpreted later.

### Loading Dataset into MLX Data Buffer

- **Buffer Creation**: The dataset is loaded into an MLX Data buffer using `dx.buffer_from_vector(dataset)`. This buffer serves as an efficient, indexable container for the image samples.

### Streamlit App Interface

- **Title Rendering**: The Streamlit app's title is displayed.
- **Random Sampling Button**: A button triggers the random sampling of portraits.

I appreciate the utility of Jupyter Notebooks for experimentation purposes, but I hold reservations about their effectiveness in teaching coding. This is where my preference for Streamlit shines through. Streamlit encourages active coding, where you're typing in the code yourself, engaging more deeply with the learning process. This approach stands in contrast to the more passive experience of navigating a Jupyter Notebook, often reduced to a routine of pressing SHIFT+ENTER to advance through pre-written code cells. In essence, Streamlit fosters a more hands-on and involved learning experience.
  
### Stream Processing

Upon clicking the 'Random Sample Portraits' button:

- **Stream Initialization**:
  - `.load_image("image")`: Loads images from their paths.
  - `.image_random_h_flip("image", prob=0.5)`: Randomly flips images horizontally with a 50% probability, enhancing dataset diversity.
  - `.shuffle()`: Randomizes the order of samples in the buffer.
  - `.to_stream()`: Converts the buffer into a stream for iteration.
  - `.batch(1)`: Organizes data into batches (of size 1 in this case).
  - `.prefetch(8, 4)`: Prefetches data for efficiency, reducing I/O wait times.

### Display Logic in Streamlit

- **Grid Display**: Images are shown in a 4x1 grid for 5 rows.
- **Label Translation and Display**: Converts numeric labels to names for display.
- **Exception Handling**: Includes a `StopIteration` check to handle cases where the stream runs out of data.

### Points of Caution

- **Path Encoding**: Encoding file paths as ASCII bytes is a critical step, but it can also be a potential pitfall if not managed carefully. It's essential to ensure that these paths are properly decoded back into strings when you need to access them. However, if your workflow doesn't require extracting or directly handling these paths, this complexity can be safely set aside.
- **File Format Assumptions**: The application assumes all images are PNGs. Any deviation needs adjustment in the code.
- **I/O Errors**: MLX Data can occasionally generate I/O errors, often stemming from issues like incorrect file paths or files that are not accessible. It's crucial to identify and rectify these errors to ensure smooth data processing.

Consider this segment from the `portraits_and_labels` function:

```python
    return [
        {
            "image": str(p).encode("ascii"),  # Use full path
            "label": category_map[p.parent.name]
        }
        for p in images
    ], len(categories)
```

In this function, we're ensuring the use of absolute paths by appending the root `./portraits` to each image file path, thereby avoiding path-related errors.

However, if you were to follow the pattern shown in the Official Documentation, where only relative paths are encoded:

```python
    return [
        {
            "image": str(p.relative_to(root)).encode("ascii"),
            "category": c,
            "label": category_map[c]
        }
        for c, p in zip(categories, images)
    ]
```

You might encounter the aforementioned I/O error. This is because MLX Data expects the paths to be appropriately formatted, and often absolute paths are required to correctly locate the files. Therefore, it's important to adjust the path handling based on your specific application context and the requirements of MLX Data.

This short application not only demonstrates image data handling in MLX Data but also exemplifies the practical implementation of data augmentation and efficient data streaming techniques in a real-world scenario.

## What's Next for Menny? More Adventures Await!

Oh, what a ride it's been with MLX Data! It's been like a theme park rollercoaster – exhilarating, a bit unpredictable, and yes, I'm sporting a couple of metaphorical bruises from those unexpected twists and turns. But hey, what's an adventure without a little excitement, right?

As we close this chapter, it's not the end of the road for Menny. Oh no, we're just getting started. After I fuel up with some delicious food and catch some Z's, we'll be ready to embark on the next leg of our journey.

Get ready for the grand reveal: _Menny, the Image Wrangler_, is about to level up to become _Menny, the Face Detector_! We've got some exciting times ahead, so make sure to stick around. Menny's next chapter is just around the corner, and it promises to be a fascinating one!

Stay tuned, and let's see what amazing things Menny will accomplish next! 🌟🚀

