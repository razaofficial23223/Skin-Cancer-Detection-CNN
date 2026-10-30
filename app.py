
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Skin Image Classifier", page_icon="🔬")

st.title("Skin Image Classification")
st.warning("Educational project only. Not for medical diagnosis.")

model = tf.keras.models.load_model("skin_cancer_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)[0]

    st.subheader("Educational Model Output")
    for name, score in zip(class_names, predictions):
        st.write(f"{name}: {score * 100:.2f}%")
