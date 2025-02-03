# NEURAL-STYLE-TRANSFER

*COMPANY*: CODETECH IT SOLUTION

*NAME*: SHARVANI SURAJ MAHADIK

*INTERN ID*:CT08LKR

*DOMAIN*:ARTIFICIAL INTELLIGENCE

*DURATION*:4 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION*: 

Neural Style Transfer: A Deep Dive into Artistic Image Transformation

Neural Style Transfer (NST) is a fascinating artificial intelligence technique that allows users to take two distinct images—one representing the content and the other representing the style—and combine them to create a new image that retains the content’s structure while adopting the style of the second image. This technique, powered by deep learning, enables the transformation of ordinary photographs into stunning artworks that mimic the styles of famous painters, abstract art, or any style image.

This application leverages the power of TensorFlow and TensorFlow Hub, using a pre-trained model to implement the Neural Style Transfer technique in an easy-to-use interface built with Gradio.

Neural Style Transfer Working
Neural Style Transfer uses convolutional neural networks (CNNs) to extract features from both the content and style images. The basic idea is to preserve the content from the content image (i.e., the objects and structures within the image) while transferring the stylistic features (such as color schemes, textures, and patterns) from the style image onto the content.

The steps involved are:

Content Image: The content image is passed through the neural network to capture high-level features that define the content’s structure (objects, shapes, etc.).
Style Image: The style image is also passed through the neural network to extract the stylistic features (textures, colors, and patterns).
Stylized Image: The network then combines these two aspects, optimizing the content image to match the content structure while incorporating the style from the style image.
The entire process is handled by a pre-trained TensorFlow model that has already learned to perform style transfer efficiently.

The Code: How the Application Works

Install Required Libraries:
!pip install gradio tensorflow tensorflow_hub
This line installs the necessary libraries:

gradio: A Python library for building user interfaces that make machine learning models accessible to users via a web interface.
tensorflow: A machine learning framework used to build, train, and deploy models.
tensorflow_hub: A library that allows you to easily use pre-trained models, such as the one for Neural Style Transfer.

Importing Required Libraries:
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import gradio as gr
from PIL import Image

libraries imported:
tensorflow (as tf): Provides functions and tools for deep learning.
tensorflow_hub (as hub): Used to load the pre-trained model for Neural Style Transfer from TensorFlow Hub.
numpy (as np): A package for array operations, used to process and manipulate images.
cv2: A computer vision library, though it seems not directly used in this particular script.
gradio (as gr): Provides tools to create a web interface for users to interact with the machine learning model.
PIL.Image: Part of the Python Imaging Library, used for working with images.

Loading Pre-trained Model:

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
Here, a pre-trained model for Neural Style Transfer is loaded from TensorFlow Hub. This model performs style transfer by taking a content image and a style image as inputs and generating an output image that blends the content and style.

Image Preprocessing:

def process_image(image):
    image = image.convert("RGB")  # Convert the image to RGB format
    image = np.array(image)  # Convert image to a numpy array
    image = image.astype(np.float32)[np.newaxis, ...] / 255.0  # Normalize the image (to range [0, 1])
    return tf.image.resize(image, (256, 256))  # Resize image to 256x256
This function processes the input image before feeding it into the model:

The image is first converted to RGB format to ensure consistency.
The image is then converted to a NumPy array, which is required for processing.
The pixel values are normalized by dividing by 255.0 so that they lie in the range [0, 1] (the model expects input in this range).
The image is resized to a uniform size of 256x256 pixels, as the pre-trained model expects this size.

Applying Style Transfer:
def stylize_image(content, style):
    content = process_image(content)  # Process the content image
    style = process_image(style)  # Process the style image
    stylized_image = model(tf.constant(content), tf.constant(style))[0]  # Apply style transfer
    return Image.fromarray(np.array(stylized_image[0] * 255, dtype=np.uint8))  # Convert to an image and return
This function takes two arguments: the content image and the style image.
It first processes both images using the process_image function to ensure they're in the proper format.
The model is called with the processed content and style images. The model returns a stylized image as a tensor, which is then converted to a NumPy array and scaled back to the [0, 255] range.
Finally, it converts the resulting array into a PIL Image that can be displayed.

Generating the Stylized Image:
def generate_stylized_image(content, style):
    return stylize_image(content, style)  # Call the stylize_image function and return the result
This function serves as a wrapper around the stylize_image function. It simplifies the flow and can be used in the Gradio interface to process the user-uploaded images and apply the style transfer.

Creating the Gradio Interface:
iface = gr.Interface(
    fn=generate_stylized_image,  # Function that processes images
    inputs=[gr.Image(type="pil", label="Content Image"), gr.Image(type="pil", label="Style Image")],  # Input components
    outputs="image",  # Output component (a stylized image)
    title="Neural Style Transfer",  # Title for the app
    description="Upload a content image and a style image to generate a stylized image.",  # Description for the app
    theme="huggingface",  # Use Huggingface theme for the interface
    css=".gradio-container {background-color: #ff00ff;}"  # Apply a magenta background color
)

This block of code creates the Gradio interface:
The function generate_stylized_image is passed as the fn argument, which will be called when the user interacts with the interface.
The inputs are defined as two gr.Image components. One for the content image and one for the style image. These components allow users to upload images.
The outputs define that the result will be an image (the stylized output).
A title and description are provided to guide the user on what the application does.
The theme argument sets the Hugging Face theme for a clean and modern interface.
Custom CSS is used to set the background color of the interface to magenta (#ff00ff), providing a unique visual experience.

Launching the Interface:

iface.launch()
This command launches the Gradio interface, making it available in the browser. Users can now interact with the application, upload their images, and receive the stylized output.

Conclusion:
This code sets up a simple yet powerful Neural Style Transfer web application where users can experiment with turning their photos into stylized artworks. The use of a pre-trained model from TensorFlow Hub, combined with Gradio's intuitive interface, makes it easy to apply advanced deep learning techniques to everyday images without needing any prior machine learning knowledge.

*OUTPUT*:

![Image](https://github.com/user-attachments/assets/b37085c2-4d9a-4b0d-8d51-91a8a1e79c58)
![Image](https://github.com/user-attachments/assets/b22a0316-16f3-47c7-80b8-c6776e82482f)

