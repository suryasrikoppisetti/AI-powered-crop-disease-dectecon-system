import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load the trained model
model_path = "/content/drive/MyDrive/under leaf spots/crop_disease_model.keras"  # Update path
model = tf.keras.models.load_model(model_path)

# Load symptoms and remedies data
def load_text_file(file_path):
    disease_info = {}
    with open(file_path, "r") as file:
        for line in file.readlines():
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                disease_info[parts[0].strip()] = parts[1].strip()
    return disease_info

# Update file paths
symptoms_file = "/content/drive/MyDrive/under leaf spots/final dataset/dataset/symptoms.txt"
remedies_file = "/content/drive/MyDrive/under leaf spots/final dataset/dataset/remedies.txt"

symptoms_data = load_text_file(symptoms_file)
remedies_data = load_text_file(remedies_file)

# Function to predict crop disease
def predict_disease(image_path):
    img_size = (150, 150)  # Ensure it matches the model input size

    # Load and preprocess image
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    
    # Get class labels from symptoms file
    class_labels = list(symptoms_data.keys())  # Using symptom file keys as labels
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class

# Streamlit UI
def main():
    st.title("🌿 Crop Disease Detection App")
    st.write("Upload an image to detect the disease and get remedies!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Save uploaded image
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict disease
        disease_name = predict_disease(image_path)

        # Retrieve symptoms and remedies
        symptoms = symptoms_data.get(disease_name, "No symptoms found")
        remedies = remedies_data.get(disease_name, "No remedies found")

        # Display results
        st.subheader(f"🦠 Disease Identified: {disease_name}")
        st.write(f"📝 Symptoms: {symptoms}")
        st.write(f"💊 Remedies: {remedies}")

        st.success("✅ Analysis Completed")

if __name__ == "__main__":
    main()
