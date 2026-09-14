import os
import json

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from disease_info import get_disease_info


# ============================================================
# 1. Create FastAPI application
# ============================================================

app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using MobileNetV2",
    version="1.0.0"
)


# ============================================================
# 2. Enable CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. File paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)


# ============================================================
# 4. Load class names
# ============================================================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"class_names.json not found: {CLASS_NAMES_PATH}"
    )

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


# ============================================================
# 5. Load trained model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = load_model(MODEL_PATH)


# ============================================================
# 6. Basic configuration
# ============================================================

IMAGE_SIZE = (224, 224)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# 7. Home endpoint
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Plant Disease Detection API is running",
        "model": "MobileNetV2",
        "classes": len(class_names),
        "endpoints": {
            "health": "/health",
            "prediction": "/predict"
        }
    }


# ============================================================
# 8. Health check endpoint
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "number_of_classes": len(class_names)
    }


# ============================================================
# 9. Image preprocessing function
# ============================================================

def preprocess_image(image_bytes):
    """
    Convert uploaded image bytes into a model-ready image.
    """

    # Convert bytes to NumPy array
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # Decode image using OpenCV
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    # Check if OpenCV successfully decoded image
    if image is None:
        raise ValueError(
            "Unable to read the uploaded image."
        )

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to MobileNetV2 input size
    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # Convert to float32
    image = image.astype(
        np.float32
    )

    # MobileNetV2 preprocessing
    image = preprocess_input(
        image
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# 10. Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check file extension
    # --------------------------------------------------------

    filename = file.filename or ""

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload JPG, JPEG, or PNG."
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    # --------------------------------------------------------
    # Process image
    # --------------------------------------------------------

    try:

        processed_image = preprocess_image(
            image_bytes
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {str(error)}"
        )

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        predictions = model.predict(
            processed_image,
            verbose=0
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_index]
    )

    disease_name = class_names[predicted_index]
    # Get plant name
    plant_name = disease_name.split("___")[0]
    #  Create a user-friendly disease name
    if "___" in disease_name:
        display_disease = disease_name.split("___", 1)[1]
    else:
        display_disease = disease_name
    # Replace underscores with spaces
    display_disease = display_disease.replace("_", " ")

    # --------------------------------------------------------
    # Determine healthy/diseased status
    # --------------------------------------------------------

    if "healthy" in disease_name.lower():

        status = "Healthy"

    else:

        status = "Diseased"

    # --------------------------------------------------------
    # Get treatment and prevention
    # --------------------------------------------------------

    information = get_disease_info(
        disease_name
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "success": True,
        "plant": plant_name,
        "disease": display_disease,
        "class_name": disease_name,
        "status": status,
        "confidence": round(
            confidence * 100,
            2
        ),
        "treatment": information["treatment"],
        "prevention": information["prevention"]
    }