import streamlit as st
import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models

# Constants
MODEL_PATH = "skin_disease_model.h5"
IMG_SIZE = 64
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

@st.cache_resource
def load_skin_model():
    try:
        model = models.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(8, activation='softmax')
        ])
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_skin_model()

# Updated class labels based on your dataset
CLASS_LABELS = [
    "Cellulitis", 
    "Impetigo", 
    "Athlete's Foot", 
    "Nail Fungus", 
    "Ringworm", 
    "Cutaneous Larva Migrans", 
    "Chickenpox", 
    "Shingles"
]

# Function to predict the disease
def predict_disease(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index]) * 100
    return CLASS_LABELS[class_index], confidence

# Function to find doctors using Google Places API
def find_doctors():
    if not API_KEY:
        return ["Set GOOGLE_API_KEY environment variable to enable doctor search"]
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=dermatologists+near+Hyderabad&key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            if status == "OK":
                results = data.get("results", [])
                if results:
                    return [
                        f"{place['name']} - {place.get('formatted_address', '')}"
                        for place in results[:5]
                    ]
                else:
                    return ["No dermatologists found."]
            else:
                return [f"Search unavailable (API status: {status})"]
        else:
            return ["Failed to fetch data from Google Places API."]
    except Exception as e:
        return [f"Doctor search unavailable: {str(e)}"]

# Streamlit app
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>
        🩺 Skin Disease Classifier
    </h1>
    <p style='text-align: center; color: #7f8c8d; font-size: 1.1em;'>
        Upload a skin image for AI-powered analysis and recommendations
    </p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "Upload an image of your skin condition",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing your skin image..."):
                if model is None:
                    st.error("Model is not loaded. Cannot proceed.")
                else:
                    disease, confidence = predict_disease(image)

                    # Results
                    st.subheader(f"📋 Diagnosis Result")
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                            padding: 20px; border-radius: 15px; color: white; text-align: center;'>
                            <h2>{disease}</h2>
                            <p style='font-size: 1.2em;'>Confidence: {confidence:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Treatment suggestions
                    st.subheader("💊 Treatment Suggestions")
                    measures = {
                        "Cellulitis": "Seek medical attention immediately. Use prescribed antibiotics and keep the area clean. Rest and elevate the affected area.",
                        "Impetigo": "Apply topical antibiotics (mupirocin). Keep the area clean and covered. Avoid scratching to prevent spreading.",
                        "Athlete's Foot": "Use antifungal creams (clotrimazole, terbinafine). Keep feet dry, change socks regularly, and wear breathable shoes.",
                        "Nail Fungus": "Use antifungal nail treatments. Keep nails trimmed and dry. Consider oral medication for persistent cases.",
                        "Ringworm": "Use antifungal creams (clotrimazole, miconazole). Keep the area clean and dry. Wash bedding and clothing regularly.",
                        "Cutaneous Larva Migrans": "Use antiparasitic treatments as prescribed. Keep the area clean. Consult a doctor for oral medication if needed.",
                        "Chickenpox": "Stay hydrated. Use calamine lotion for itching. Take antiviral medications if prescribed. Avoid scratching to prevent scarring.",
                        "Shingles": "Use antiviral medications (acyclovir). Take pain relievers as needed. Apply cool compresses. Rest and avoid stress."
                    }
                    st.info(measures.get(disease, "Consult a healthcare professional for proper diagnosis."))

                    # Disclaimer
                    st.warning("""
                        ⚠️ **Disclaimer**: This is an AI-powered screening tool for educational purposes only. 
                        It is NOT a substitute for professional medical advice. Always consult a qualified 
                        dermatologist for accurate diagnosis and treatment.
                    """)

                    # Doctor recommendations
                    st.subheader("🏥 Nearby Dermatologists")
                    doctors = find_doctors()
                    for i, doctor in enumerate(doctors, 1):
                        st.write(f"{i}. {doctor}")
    else:
        st.info("👆 Upload an image to get started!")
