import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import json
import sys
from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "backend" / "model" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "backend" / "model" / "class_names.json"

BACKEND_PATH = BASE_DIR / "backend"

if str(BACKEND_PATH) not in sys.path:
    sys.path.append(str(BACKEND_PATH))

from disease_info import get_disease_info


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 45px;
        font-weight: 700;
        text-align: center;
        color: #198754;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 35px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        background-color: #f1f8f3;
        margin-top: 20px;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e5e5e5;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: #777;
        margin-top: 50px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(uploaded_file):

    image_bytes = uploaded_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (224, 224)
    )

    # Float32
    image = image.astype(np.float32)

    # MobileNetV2 preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_disease(uploaded_file):

    model = load_model()
    class_names = load_class_names()

    image = preprocess_image(
        uploaded_file
    )

    predictions = model.predict(
        image,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_index]
    )

    class_name = class_names[predicted_index]

    # Extract plant name
    if "___" in class_name:
        plant_name = class_name.split("___")[0]
        disease_name = class_name.split("___", 1)[1]
    else:
        plant_name = class_name
        disease_name = class_name

    # Make disease name readable
    disease_name = disease_name.replace(
        "_",
        " "
    )

    # Determine status
    if "healthy" in class_name.lower():
        status = "Healthy"
    else:
        status = "Diseased"

    # Disease information
    information = get_disease_info(
        class_name
    )

    return {
        "plant": plant_name,
        "disease": disease_name,
        "class_name": class_name,
        "status": status,
        "confidence": confidence * 100,
        "treatment": information["treatment"],
        "prevention": information["prevention"]
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌿 Plant Disease Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered plant disease detection using MobileNetV2</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 Navigation")

    page = st.radio(
        "Select Page",
        [
            "Disease Detection",
            "About",
            "How It Works"
        ]
    )


# ============================================================
# DISEASE DETECTION PAGE
# ============================================================

if page == "Disease Detection":

    st.header("🔍 Detect Plant Disease")

    st.write(
        "Upload an image of a plant leaf to identify the possible disease."
    )

    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.subheader("📷 Image Preview")

        st.image(
            uploaded_file,
            caption="Uploaded Plant Leaf",
            width=400
        )

        st.write("")

        if st.button(
            "🔍 Detect Disease",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                try:

                    result = predict_disease(
                        uploaded_file
                    )

                    st.success(
                        "Prediction completed successfully!"
                    )

                    # =========================================
                    # RESULT
                    # =========================================

                    st.markdown(
                        '<div class="result-card">',
                        unsafe_allow_html=True
                    )

                    st.subheader(
                        "🌿 Prediction Result"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Plant",
                            result["plant"]
                        )

                        st.metric(
                            "Disease",
                            result["disease"]
                        )

                    with col2:

                        st.metric(
                            "Status",
                            result["status"]
                        )

                        st.metric(
                            "Confidence",
                            f'{result["confidence"]:.2f}%'
                        )

                    st.progress(
                        min(
                            int(result["confidence"]),
                            100
                        )
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                    # =========================================
                    # TREATMENT
                    # =========================================

                    st.markdown(
                        '<div class="info-card">',
                        unsafe_allow_html=True
                    )

                    st.subheader(
                        "💊 Treatment"
                    )

                    st.write(
                        result["treatment"]
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                    # =========================================
                    # PREVENTION
                    # =========================================

                    st.markdown(
                        '<div class="info-card">',
                        unsafe_allow_html=True
                    )

                    st.subheader(
                        "🛡️ Prevention"
                    )

                    st.write(
                        result["prevention"]
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {str(e)}"
                    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.header("📖 About the Project")

    st.write(
        """
        Plant Disease Detection System is an AI-based application
        designed to identify diseases from plant leaf images.

        The system uses a MobileNetV2 deep learning model trained
        using transfer learning. The model can classify images into
        38 different plant disease categories.
        """
    )

    st.subheader("🧠 Model")

    st.write(
        """
        • MobileNetV2

        • ImageNet pretrained weights

        • Input size: 224 × 224 × 3

        • 38 disease classes

        • TensorFlow / Keras
        """
    )

    st.subheader("📊 Model Performance")

    st.write(
        """
        Initial test accuracy: approximately 94.67%
        """
    )


# ============================================================
# HOW IT WORKS PAGE
# ============================================================

elif page == "How It Works":

    st.header("⚙️ How It Works")

    st.markdown(
        """
        ### 1️⃣ Upload Image

        The user uploads a JPG, JPEG or PNG image of a plant leaf.

        ### 2️⃣ Image Preprocessing

        The image is processed using OpenCV.

        The image is:

        - Converted from BGR to RGB
        - Resized to 224 × 224 pixels
        - Converted to float32
        - Processed using MobileNetV2 preprocessing

        ### 3️⃣ Disease Prediction

        The processed image is passed to the trained MobileNetV2 model.

        ### 4️⃣ Classification

        The model predicts one of the 38 available classes.

        ### 5️⃣ Result

        The application displays:

        - Plant name
        - Disease name
        - Disease status
        - Confidence score
        - Treatment
        - Prevention
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌿 Plant Disease Detection System<br>
        AI & Machine Learning Academic Project
    </div>
    """,
    unsafe_allow_html=True
)