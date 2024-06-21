import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.applications.EfficientNetB0(weights="imagenet")
decode_predictions = tf.keras.applications.efficientnet.decode_predictions
preprocess_input = tf.keras.applications.efficientnet.preprocess_input




def load_image(image_file):
    img = Image.open(image_file)
    return img

def analyze_image(image):
    img_array = np.array(image.resize((224, 224)))  # Resize image to the input size of the model
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image
    predictions = model.predict(img_array)  # Make predictions
    decoded_predictions = decode_predictions(predictions, top=10)[0]  # Decode the predictions
    return decoded_predictions

# Streamlit user interface
st.title("Image Component Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        st.write("Analyzing...")
        components = analyze_image(image)
        st.write("Detected components:")
        for i, (imagenet_id, label, score) in enumerate(components):
            st.write(f"{i+1}. {label} ({score*100:.2f}%)")
