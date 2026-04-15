import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import requests

# Constants
MODEL_PATH = "skin_disease_model.h5"
IMG_SIZE = 64
API_KEY = "AIzaSyAY9l3JSbXCAH4ygyn8ccBu6-OXzbms57s"  # Ensure this is your valid API key

# Load the trained model with robust fallback
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras import models, layers

# Fix InputLayer issues
def custom_input_layer(**config):
    config.pop("batch_shape", None)
    config.pop("batch_input_shape", None)
    config.pop("ragged", None)
    config.pop("sparse", None)
    return InputLayer(**config)

# Fix DTypePolicy issue
def custom_dtype_policy(**config):
    return Policy("float32")  # force safe default


def build_model():
    m = models.Sequential([
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
    return m


model = None
model_error = None
try:
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "InputLayer": custom_input_layer,
            "DTypePolicy": custom_dtype_policy
        }
    )
except Exception as e:
    model_error = e
    try:
        model = build_model()
        model.load_weights(MODEL_PATH)
    except Exception as ew:
        model_error = ew

if model is None:
    raise RuntimeError(f"Cannot load model from {MODEL_PATH}: {model_error}")

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
def predict_disease(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return CLASS_LABELS[class_index]

# Function to find doctors using Google Places API
def find_doctors():
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=dermatologists+in+India&key={API_KEY}"
    response = requests.get(url)
    
    # Check if the response was successful
    if response.status_code == 200:
        data = response.json()
        status = data.get("status")
        if status == "OK":
            results = data.get("results", [])
            if results:
                doctor_names = [place["name"] for place in results[:5]]
                return doctor_names
            else:
                return ["No dermatologists found."]
        else:
            error_message = data.get("error_message", "API request denied.")
            return [f"API Error: {error_message}"]
    else:
        return ["Failed to fetch data from Google Places API."]

# Streamlit app
st.set_page_config(page_title="Skin Disease Classifier", layout="wide")
st.title("Skin Disease Classifier")

# Upload file
uploaded_file = st.file_uploader("Upload an image of your skin condition", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image("temp_image.jpg", caption="Uploaded Image", use_column_width=True)
    if st.button("Analyze"):
        disease = predict_disease("temp_image.jpg")
        st.subheader(f"Diagnosis: {disease}")
        
        # General measures
        st.write("**General Measures:**")
        measures = {
            "Cellulitis": "Seek medical attention and use prescribed antibiotics.",
            "Impetigo": "Apply topical antibiotics and keep the area clean.",
            "Athlete's Foot": "Keep your feet dry and use antifungal cream.",
            "Nail Fungus": "Use antifungal treatments for nails.",
            "Ringworm": "Use antifungal creams or oral medications.",
            "Cutaneous Larva Migrans": "Use antiparasitic treatments as prescribed.",
            "Chickenpox": "Stay hydrated and use antiviral medications.",
            "Shingles": "Use antiviral medications and pain relief treatments."
        }
        st.write(measures[disease])

        # Recommend doctors
        st.subheader("Recommended Dermatologists:")
        doctors = find_doctors()
        for doctor in doctors:
            st.write(f"- {doctor}")
