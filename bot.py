from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model_path = "./oral_cancer_classification_model.h5"
model = tf.keras.models.load_model(model_path)

# Parameters
img_height, img_width = 224, 224  # Same dimensions used during training

def preprocess_image(image):
    # Convert image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Resize the image to match model input
    image = image.resize((img_height, img_width))
    # Convert image to array and normalize pixel values
    img_array = img_to_array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_image(image):
    # Preprocess the image
    img_array = preprocess_image(image)
    # Get the prediction
    prediction = model.predict(img_array)
    # Interpret the prediction
    class_label = "Cancerous" if prediction[0][0] < 0.5 else "Normal"
    confidence = prediction[0][0] if class_label == "Normal" else 1 - prediction[0][0]
    return class_label, confidence

# Streamlit app
def main():
    st.title("Oral Cancer Classification")
    st.write("Upload an image to classify it as 'Cancerous' or 'Normal'.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform prediction
        with st.spinner("Classifying the image..."):
            label, confidence = predict_image(image)
        
        # Display results
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}")
        
        # Optionally, add more details
        if label == "Cancerous":
            st.warning("This image is classified as Cancerous. Consult a medical professional for further advice.")
        else:
            st.info("This image is classified as Normal.")

if __name__ == "__main__":
    main()
