import streamlit as st
import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras import layers, models

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
        model.compile()
        return model
    except Exception:
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH, compile=False)
            return model
        except Exception as e2:
            st.error(f"Could not load model: {e2}")
            return None

model = load_skin_model()

CLASS_LABELS = [
    "Cellulitis", "Impetigo", "Athlete's Foot", "Nail Fungus",
    "Ringworm", "Cutaneous Larva Migrans", "Chickenpox", "Shingles"
]

def predict_disease(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index]) * 100
    return CLASS_LABELS[class_index], confidence

def find_doctors():
    if not API_KEY:
        return ["Set GOOGLE_API_KEY env var to enable doctor search"]
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=dermatologists+near+Hyderabad&key={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "OK":
                results = data.get("results", [])
                return [f"{r['name']} - {r.get('formatted_address', '')}" for r in results[:5]] or ["No results"]
            return [f"API: {data.get('status')}"]
        return [f"HTTP {resp.status_code}"]
    except Exception as e:
        return [f"Search error: {e}"]

st.set_page_config(page_title="Skin Disease Classifier", page_icon="🔬", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#2c3e50;'>🩺 Skin Disease Classifier</h1>
<p style='text-align:center;color:#7f8c8d;font-size:1.1em;'>Upload a skin image for AI analysis</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded", use_container_width=True)
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                if model is None:
                    st.error("Model not loaded. Check logs.")
                else:
                    disease, conf = predict_disease(image)
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;
                    border-radius:15px;color:white;text-align:center;'>
                    <h2>{disease}</h2><p style='font-size:1.2em;'>Confidence: {conf:.1f}%</p></div>
                    """, unsafe_allow_html=True)
                    measures = {
                        "Cellulitis": "Seek medical attention. Use prescribed antibiotics.",
                        "Impetigo": "Apply topical antibiotics. Keep area clean.",
                        "Athlete's Foot": "Use antifungal cream. Keep feet dry.",
                        "Nail Fungus": "Use antifungal nail treatment.",
                        "Ringworm": "Use antifungal cream. Keep area dry.",
                        "Cutaneous Larva Migrans": "Use antiparasitic treatment as prescribed.",
                        "Chickenpox": "Stay hydrated. Use calamine lotion.",
                        "Shingles": "Use antivirals. Apply cool compresses."
                    }
                    st.info(f"💊 {measures.get(disease, 'Consult a doctor.')}")
                    st.warning("⚠️ Educational tool only. Not medical advice.")
                    st.subheader("🏥 Nearby Dermatologists")
                    for i, d in enumerate(find_doctors(), 1):
                        st.write(f"{i}. {d}")
    else:
        st.info("👆 Upload to start")
