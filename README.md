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

How Does Neural Style Transfer Work?
Neural Style Transfer uses convolutional neural networks (CNNs) to extract features from both the content and style images. The basic idea is to preserve the content from the content image (i.e., the objects and structures within the image) while transferring the stylistic features (such as color schemes, textures, and patterns) from the style image onto the content.

The steps involved are:

Content Image: The content image is passed through the neural network to capture high-level features that define the content’s structure (objects, shapes, etc.).
Style Image: The style image is also passed through the neural network to extract the stylistic features (textures, colors, and patterns).
Stylized Image: The network then combines these two aspects, optimizing the content image to match the content structure while incorporating the style from the style image.
The entire process is handled by a pre-trained TensorFlow model that has already learned to perform style transfer efficiently.

The Code: How the Application Works
Here’s how the code is structured and how each part contributes to the functioning of the application:

1. Installing Libraries
Before we begin, we need to install the necessary libraries:

!pip install gradio tensorflow tensorflow_hub

This command installs the Gradio library for the interface, TensorFlow for running deep learning models, and TensorFlow Hub for loading the pre-trained model.

2. Loading the Pre-trained Model
The neural style transfer model is loaded from TensorFlow Hub, which provides various pre-trained models that can be easily accessed and integrated into applications:

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

This particular model, arbitrary-image-stylization-v1-256, is capable of transferring the style of one image onto another, using deep learning techniques.

3. Processing the Images
Before applying the style transfer, we need to process the input images (both content and style images). This involves resizing them and normalizing the pixel values for the model:

def process_image(image):
    image = image.convert("RGB")  # Convert the image to RGB format
    image = np.array(image)  # Convert image to a numpy array
    image = image.astype(np.float32)[np.newaxis, ...] / 255.0  # Normalize the image
    return tf.image.resize(image, (256, 256))  # Resize to 256x256

This function ensures that both images are in the proper format and size for the neural network to process.

4. Applying the Style Transfer
The core of the application lies in the function that performs the neural style transfer:

def stylize_image(content, style):
    content = process_image(content)
    style = process_image(style)
    stylized_image = model(tf.constant(content), tf.constant(style))[0]
    return Image.fromarray(np.array(stylized_image[0] * 255, dtype=np.uint8))

    Content and Style Processing: Both the content and style images are processed using the process_image function.
Neural Style Transfer: The pre-trained model is then called with both processed images, and it returns the stylized image.
Returning the Result: The output is converted back to a format suitable for display (PIL Image), ensuring that the image is scaled to the proper range.

5. Creating the Gradio Interface
The Gradio interface is designed to be interactive and user-friendly. Users can upload their content and style images and view the results instantly:

iface = gr.Interface(
    fn=generate_stylized_image,
    inputs=[gr.Image(type="pil", label="Content Image"), gr.Image(type="pil", label="Style Image")],
    outputs="image",
    title="Neural Style Transfer",
    description="Upload a content image and a style image to generate a stylized image.",
    theme="huggingface",  # Use Huggingface theme for UI
    css=".gradio-container {background-color: #ff00ff;}"  # Set magenta background color
)

This block of code:

Defines the function to be called when images are uploaded (generate_stylized_image).
Specifies the inputs (content and style images) and the output (the stylized image).
Customizes the UI using Gradio's built-in theme and applies a magenta background color using CSS.
The title and description provide users with context and instructions for using the application.

6. Launching the Interface
Finally, we launch the interface:

iface.launch()

This command starts the Gradio interface, allowing users to interact with the app via their browser. Once the app is live, users can upload images and immediately see the results of the style transfer.

Features of the Application
User-Friendly Interface: The application’s Gradio interface is designed to be simple and intuitive. Users just need to upload two images—one for content and one for style—and the system automatically handles the rest.
Magenta Background: The interface features a vibrant magenta background, creating a visually appealing environment that enhances the user experience.
Hugging Face Theme: The application uses the Hugging Face theme for a clean and modern look, ensuring that users can easily navigate the interface.
Seamless Image Processing: The combination of TensorFlow, TensorFlow Hub, and Gradio makes the style transfer process fast and seamless, even for high-resolution images.
Practical Applications of Neural Style Transfer
Neural Style Transfer has numerous applications:

Photo Editing: Artists and photographers can use this technique to apply custom styles to their photos, creating unique, artistic effects.
Design: Graphic designers can use style transfer to generate eye-catching visuals for advertisements, posters, and websites.
Entertainment: The film and gaming industry uses NST to create stylized visuals and concept art.
Personalized Art: Individuals can use this tool to turn their personal photos into beautiful pieces of art, making them suitable for use as gifts or home decor.

Conclusion
The Neural Style Transfer application provides an easy way for anyone to turn their photos into beautiful, stylized artworks. By combining the power of TensorFlow's deep learning model with the user-friendly Gradio interface, this app allows you to experiment with artistic styles and create something unique with just a few clicks. Whether you are a professional artist, a hobbyist, or someone who simply enjoys experimenting with AI, this tool provides endless creative possibilities to explore the fusion of art and technology.

*OUTPUT*:

![Image](https://github.com/user-attachments/assets/b37085c2-4d9a-4b0d-8d51-91a8a1e79c58)
![Image](https://github.com/user-attachments/assets/b22a0316-16f3-47c7-80b8-c6776e82482f)

