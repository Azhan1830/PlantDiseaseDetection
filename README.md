# 🌿 Plant Disease Detection System

An AI-based Plant Disease Detection System that identifies plant diseases from leaf images using deep learning.

The system uses MobileNetV2 with TensorFlow/Keras for image classification and FastAPI for the backend API. It also provides Streamlit and HTML/CSS/Bootstrap interfaces for interacting with the prediction system.

---

## 📌 Project Overview

Plant diseases can reduce crop quality and production. Early detection can help farmers and agricultural professionals take appropriate action.

This project uses a deep learning model to classify plant leaf images into different plant disease categories. The trained model can recognize 38 different classes.

---

## 🚀 Technologies Used

### Machine Learning
* Python
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* NumPy

### Backend
* FastAPI
* Uvicorn
* Python

### Frontend
* HTML
* CSS
* Bootstrap
* JavaScript

### Additional Interface
* Streamlit

### Development Tools
* Google Colab
* Visual Studio Code
* Git
* GitHub

---

## 🧠 Machine Learning Model

The project uses MobileNetV2 with ImageNet pretrained weights.

### Model Configuration
* **Input size:** 224 × 224 × 3
* **Architecture:** MobileNetV2
* **Pretrained weights:** ImageNet
* **Layers:** Global Average Pooling, Dropout, Dense classification layer, Softmax output
* **Number of classes:** 38

Transfer learning was used to adapt MobileNetV2 for plant disease classification.

---

## 📊 Dataset

The project uses a plant disease image dataset based on PlantVillage. The dataset contains images belonging to multiple plant and disease categories.

The final dataset used for the project contains:
* **Training images:** 43,443
* **Validation images:** 5,422
* **Test images:** 5,439
* **Classes:** 38

Images were resized to `224 × 224` and MobileNetV2 preprocessing was applied.

---

## 📈 Model Performance

The initial MobileNetV2 model achieved approximately:
* **Test Accuracy:** 94.67%
* **Test Loss:** 0.1581

The evaluation also included Precision, Recall, F1-score, Classification report, Confusion matrix, Training/validation accuracy, and Training/validation loss.

---

## 🏗️ System Architecture

```text
                         User
                           |
             +-------------+-------------+
             |                           |
       HTML Frontend                Streamlit
             |                           |
             +-------------+-------------+
                           |
                        FastAPI
                           |
                  Image Preprocessing
                           |
                         OpenCV
                           |
                      MobileNetV2
                           |
                      Prediction
                           |
                 Disease Information
                           |
              +------------+------------+
              |                         |
          Treatment                Prevention

---

