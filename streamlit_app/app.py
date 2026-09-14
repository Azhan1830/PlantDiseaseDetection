import streamlit as st
import requests


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# FastAPI endpoint
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 20px;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🌱 Plant Disease Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a plant leaf image and use AI to identify possible diseases.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# Navigation
# ============================================================

page = st.radio(
    "Navigation",
    ["Disease Detection", "About", "How It Works"],
    horizontal=True
)


# ============================================================
# About page
# ============================================================

if page == "About":

    st.header("About the Project")

    st.write(
        """
        The Plant Disease Detection System uses deep learning
        to identify plant diseases from leaf images.

        The system uses a MobileNetV2-based image classification
        model trained on the PlantVillage dataset.
        """
    )

    st.subheader("Technology Used")

    st.write("""
    - Python
    - TensorFlow / Keras
    - MobileNetV2
    - OpenCV
    - FastAPI
    - Streamlit
    - HTML / CSS / Bootstrap
    """)


# ============================================================
# How It Works
# ============================================================

elif page == "How It Works":

    st.header("How It Works")

    st.write("The system follows these steps:")

    st.write("""
    1. Upload a plant leaf image.
    2. The image is sent to the FastAPI backend.
    3. OpenCV processes the image.
    4. The image is resized to 224 × 224 pixels.
    5. MobileNetV2 preprocessing is applied.
    6. The trained model predicts the disease.
    7. The system calculates the confidence score.
    8. Treatment and prevention guidance are displayed.
    """)


# ============================================================
# Disease Detection
# ============================================================

else:

    st.header("Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )


    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Plant Leaf",
            use_container_width=True
        )

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        st.write(
            f"**Size:** {uploaded_file.size / 1024:.1f} KB"
        )


        # ----------------------------------------------------
        # Detect button
        # ----------------------------------------------------

        if st.button(
            "🔍 Detect Disease",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing plant image..."
            ):

                try:

                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        API_URL,
                        files=files,
                        timeout=60
                    )


                    # ----------------------------------------
                    # Successful prediction
                    # ----------------------------------------

                    if response.status_code == 200:

                        result = response.json()

                        st.success(
                            "Image analyzed successfully!"
                        )

                        st.markdown(
                            '<div class="result-box">',
                            unsafe_allow_html=True
                        )

                        st.subheader(
                            "Prediction Result"
                        )

                        # Plant
                        st.write(
                            f"🌿 **Plant:** {result['plant']}"
                        )

                        # Disease
                        st.write(
                            f"🦠 **Disease:** {result['disease']}"
                        )

                        # Status
                        if result["status"] == "Healthy":

                            st.success(
                                "✅ Plant Status: Healthy"
                            )

                        else:

                            st.error(
                                "⚠️ Plant Status: Diseased"
                            )

                        # Confidence
                        confidence = result["confidence"]

                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )

                        st.progress(
                            min(
                                confidence / 100,
                                1.0
                            )
                        )

                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )


                        # ------------------------------------
                        # Treatment
                        # ------------------------------------

                        st.markdown(
                            '<div class="section-title">'
                            '💊 Treatment'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.info(
                            result["treatment"]
                        )


                        # ------------------------------------
                        # Prevention
                        # ------------------------------------

                        st.markdown(
                            '<div class="section-title">'
                            '🛡️ Prevention'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.info(
                            result["prevention"]
                        )


                    # ----------------------------------------
                    # API error
                    # ----------------------------------------

                    else:

                        try:

                            error_detail = response.json().get(
                                "detail",
                                "Unable to process the image."
                            )

                        except Exception:

                            error_detail = (
                                "Unable to process the image."
                            )

                        st.error(
                            f"Error: {error_detail}"
                        )


                # ------------------------------------------------
                # FastAPI connection error
                # ------------------------------------------------

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Unable to connect to FastAPI. "
                        "Please start the FastAPI server first."
                    )


                # ------------------------------------------------
                # Timeout
                # ------------------------------------------------

                except requests.exceptions.Timeout:

                    st.error(
                        "The prediction request timed out. "
                        "Please try again."
                    )


                # ------------------------------------------------
                # Other errors
                # ------------------------------------------------

                except Exception as error:

                    st.error(
                        f"Unexpected error: {error}"
                    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Plant Disease Detection System • MobileNetV2 + FastAPI + Streamlit"
)

st.caption(
    "AI predictions are for educational purposes and should not "
    "replace professional agricultural advice."
)