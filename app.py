import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # Use VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pyttsx3
from PIL import Image

#import gdown
import requests
import os

# Load VGG16 model
vgg16_model = VGG16(weights="imagenet")
vgg16_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output)  # second-to-last layer for features


url = "https://www.dropbox.com/scl/fi/ziiaynarnez45qgnb76fo/Attentionmodel.h5?rlkey=z8eec4nqmmdjpdkl1u8y6nogt&st=zmglaqco&dl=1"

response = requests.get(url, stream=True)
with open("Attentionmodel.h5", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)  

model = tf.keras.models.load_model("Attentionmodel.h5")


# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app title and description
st.title("Image Caption Generator")
st.markdown(
    "Upload or capture an image, and this app will generate a caption for it using a trained LSTM model."
)

# Function to process image and generate caption
def process_image_and_generate_caption(image):
    # Resize the image for VGG16 input
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features using VGG16
    image_features = vgg16_model.predict(image, verbose=0)

    # Max caption length
    max_caption_length = 34
    
    # Define function to get word from index
    def get_word_from_index(index, tokenizer):
        return next(
            (word for word, idx in tokenizer.word_index.items() if idx == index), None
        )

    # Generate caption using the model
    def predict_caption(model, image_features, tokenizer, max_caption_length):
        caption = "startseq"
        for _ in range(max_caption_length):
            sequence = tokenizer.texts_to_sequences([caption])[0]
            sequence = pad_sequences([sequence], maxlen=max_caption_length)
            yhat = model.predict([image_features, sequence], verbose=0)
            predicted_index = np.argmax(yhat)
            predicted_word = get_word_from_index(predicted_index, tokenizer)
            caption += " " + predicted_word
            if predicted_word is None or predicted_word == "endseq":
                break
        return caption

    # Generate caption
    generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

    # Remove startseq and endseq
    generated_caption = generated_caption.replace("startseq", "").replace("endseq", "").strip()
    
    return generated_caption

# Function to convert text to speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit interface for uploading image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Streamlit interface for capturing image using webcam
image_input = st.camera_input("Or capture an image")

# Handle uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    image = load_img(uploaded_image, target_size=(224, 224))  # Resize to VGG16's input size
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Generating caption..."):
        generated_caption = process_image_and_generate_caption(image)
        st.write(f"Generated Caption: {generated_caption}")
        speak_text(generated_caption)

# Handle captured image
if image_input is not None:
    st.subheader("Captured Image")
    image = Image.open(image_input).convert("RGB")  # Convert image to RGB format
    st.image(image, caption="Captured Image", use_column_width=True)
    
    with st.spinner("Generating caption..."):
        generated_caption = process_image_and_generate_caption(image)
        st.write(f"Generated Caption: {generated_caption}")
        speak_text(generated_caption)
