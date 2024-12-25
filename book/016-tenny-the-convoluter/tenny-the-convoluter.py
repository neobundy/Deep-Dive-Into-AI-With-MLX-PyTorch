import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


# Define the Tenny model as a subclass of nn.Module
class Tenny(nn.Module):
    def __init__(self):
        super(Tenny, self).__init__()
        # Define a convolutional layer with 1 input channel (grayscale),
        # 2 output channels, and a 3x3 kernel size. Bias is False since we are manually setting filters.
        self.conv = nn.Conv2d(1, 2, 3, bias=False)

        # Manually setting the weights of the convolutional layer to Sobel filters for edge detection.
        # The first filter detects horizontal edges, and the second detects vertical edges.
        sobel_filters = torch.tensor([[[[-1., 0., 1.],
                                        [-2., 0., 2.],
                                        [-1., 0., 1.]]],
                                      [[[-1., -2., -1.],
                                        [0., 0., 0.],
                                        [1., 2., 1.]]]])
        self.conv.weight = nn.Parameter(sobel_filters)

    def forward(self, x):
        # Forward pass of the model, applying the convolutional layer to the input
        x = self.conv(x)
        return x


# Function to perform edge detection using Tenny
def tenny_edge_detection(image_path):
    # Transformation pipeline: convert image to grayscale and then to a tensor
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Open the image and apply the transformations
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension to fit model input

    # Instantiate Tenny and apply it to the image
    tenny = Tenny()
    edge_detected = tenny(image)

    # Convert the output tensor to a numpy array and scale it to 0-255 for image display
    edge_detected = edge_detected.squeeze().detach().numpy()
    edge_detected_image = np.uint8(edge_detected / edge_detected.max() * 255)

    return edge_detected_image


# Path to Tenny's avatar image
TENNY_AVATAR = './images/tenny-the-convoluter.png'

# Streamlit UI setup
st.title('Tenny, the Convoluter')

# Placeholder for Tenny's avatar or the uploaded image
avatar_placeholder = st.empty()
avatar_placeholder.image(TENNY_AVATAR, use_column_width=True)

# Sidebar for image upload
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Clear Tenny's avatar and show the uploaded image
    avatar_placeholder.empty()
    with avatar_placeholder.container():
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Detecting edges...")

    # Perform edge detection on the uploaded image
    processed_image = tenny_edge_detection(uploaded_file)

    # Display the processed (edge-detected) image
    processed_image = Image.fromarray(processed_image[0])  # Convert to PIL Image format for Streamlit
    st.image(processed_image, caption='Processed Image', use_column_width=True)
