import streamlit as st
from PIL import Image
import numpy as np

IMAGE_SCALE_FACTOR = 4


def load_image(image_path):
    """ Load an image from a file path and scale it down by a factor of IMAGE_SCALE_FACTOR. """
    img = Image.open(image_path)
    width, height = img.size
    new_size = (width // IMAGE_SCALE_FACTOR, height // IMAGE_SCALE_FACTOR)
    return img.resize(new_size)


def display_image(img, caption):
    """Display an image with a caption."""
    st.image(img, caption=caption)


def normalize_image(img):
    """Normalize an image by converting its pixel values to the range [0, 1]."""
    return np.array(img) / 255.0


def denormalize_image(img_array):
    """Denormalize an image by converting its pixel values back to the range [0, 255]."""
    return (img_array * 255).astype(np.uint8)


def adjust_brightness(img_array: np.ndarray, value: float):
    """Adjust the brightness of an image by adding a value to each pixel."""
    img_array += value

    # Clip the values to the range [0, 1] to ensure they remain valid
    np.clip(img_array, 0, 1, out=img_array)

    return img_array


def main():
    """Main function to run the application."""
    image_path = "images/a_cat.jpg"  # Path to the image
    st.title("Image Inspector")

    # Load and display the original image
    img = load_image(image_path)
    display_image(img, 'Original Image')

    # Normalize the image and display it
    normalized_img_array = normalize_image(img)
    normalized_img = Image.fromarray((normalized_img_array * 255).astype(np.uint8))
    display_image(normalized_img, 'Normalized Image')

    # Get the brightness adjustment value from the slider
    brightness_adjustment_value = st.slider("Brightness Adjustment Value", min_value=-1.0, max_value=1.0, value=0.0,
                                            step=0.1)

    # Adjust the brightness of the normalized image and display it
    adjusted_brightness_img_array = adjust_brightness(normalized_img_array, brightness_adjustment_value)
    adjusted_brightness_img = Image.fromarray((adjusted_brightness_img_array * 255).astype(np.uint8))
    display_image(adjusted_brightness_img, 'Adjusted Brightness Image')


if __name__ == "__main__":
    main()