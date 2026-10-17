
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="PlantHealth AI", page_icon="🌿")
st.title("🌿 PlantHealth AI")
st.write("Upload a leaf image to predict its disease.")

MODEL_PATH = os.path.join("models", "best_custom_cnn.keras")

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot",
    "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___healthy",
    "Cherry___Powdery_mildew", "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust", "Corn___healthy",
    "Corn___Northern_Leaf_Blight", "Grape___Black_rot",
    "Grape___Esca", "Grape___healthy", "Grape___Leaf_blight",
    "Orange___Haunglongbing", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
    "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

GUIDANCE = {
    "Potato___Early_blight": {
        "symptoms": "Older leaves may develop brown spots with concentric rings.",
        "prevention": "Remove infected plant debris, rotate crops, and avoid prolonged leaf wetness.",
        "treatment": "Confirm the disease with an agricultural expert. Use only a fungicide registered for this crop and disease, following its label."
    },
    "Potato___Late_blight": {
        "symptoms": "Dark, spreading leaf lesions may develop, especially in humid conditions.",
        "prevention": "Inspect plants regularly, avoid prolonged leaf wetness, and manage infected debris safely.",
        "treatment": "Seek prompt agricultural advice. Use only locally registered treatments according to the product label."
    },
    "Tomato___Early_blight": {
        "symptoms": "Brown leaf spots may have concentric rings, often on older leaves.",
        "prevention": "Rotate crops, manage infected debris, and water near the soil.",
        "treatment": "Confirm the disease and ask a local agricultural expert about registered treatments."
    },
    "Tomato___Late_blight": {
        "symptoms": "Dark, spreading lesions may develop on leaves and stems.",
        "prevention": "Monitor plants frequently and avoid prolonged leaf wetness.",
        "treatment": "Seek prompt agricultural advice and use only locally registered treatments according to their labels."
    }
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Check the models folder.")
    st.stop()

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Predict Disease"):
        resized = image.resize((224, 224))
        array = np.array(resized, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)

        scores = model.predict(array, verbose=0)[0]
        index = int(np.argmax(scores))
        disease = CLASS_NAMES[index]
        confidence = float(scores[index]) * 100

        st.subheader("Prediction Result")
        st.write("**Predicted class:**", disease.replace("___", " — "))
        st.write(f"**Model confidence:** {confidence:.2f}%")

        if confidence < 70:
            st.warning(
                "Low-confidence prediction. Confirm the disease "
                "before deciding on treatment."
            )

        if disease.endswith("___healthy"):
            st.success("The model predicts a healthy class.")
            st.info("Continue regular crop monitoring.")
        else:
            details = GUIDANCE.get(disease)

            if details:
                st.subheader("Possible symptoms")
                st.write(details["symptoms"])
                st.subheader("Prevention")
                st.write(details["prevention"])
                st.subheader("Treatment guidance")
                st.write(details["treatment"])
            else:
                st.info(
                    "Specific guidance is not available for this class yet. "
                    "Consult a local agricultural expert before applying pesticides."
                )

        st.caption(
            "AI predictions may be incorrect. Confirm the disease before treatment."
        )
