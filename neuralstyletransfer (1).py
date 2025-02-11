# -*- coding: utf-8 -*-
"""NeuralStyleTransfer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TiIImVQST2eVvC4XbS7J9opYORwVCmbk
"""

# Install necessary libraries
!pip install gradio tensorflow tensorflow_hub

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import gradio as gr
from PIL import Image

# Load pre-trained model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to process image
def process_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = image.astype(np.float32)[np.newaxis, ...] / 255.0
    return tf.image.resize(image, (256, 256))

# Function to apply style transfer
def stylize_image(content, style):
    content = process_image(content)
    style = process_image(style)
    stylized_image = model(tf.constant(content), tf.constant(style))[0]
    return Image.fromarray(np.array(stylized_image[0] * 255, dtype=np.uint8))

# Function to generate stylized image
def generate_stylized_image(content, style):
    return stylize_image(content, style)

# Create Gradio interface with a magenta background color
iface = gr.Interface(
    fn=generate_stylized_image,
    inputs=[gr.Image(type="pil", label="Content Image"), gr.Image(type="pil", label="Style Image")],
    outputs="image",
    title="Neural Style Transfer",
    description="Upload a content image and a style image to generate a stylized image.",
    theme="huggingface",  # Use Huggingface theme for UI
    css=".gradio-container {background-color: #ff00ff;}"  # Set magenta background color
)

# Launch the Gradio interface
iface.launch()