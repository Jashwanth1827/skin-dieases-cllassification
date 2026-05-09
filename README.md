---
title: Skin Disease Classifier
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.31.0
app_file: skin_disease_app.py
pinned: false
---

# 🧠 Skin Disease Classifier

An AI-powered web application that predicts common skin diseases from images using a Convolutional Neural Network (CNN) and provides basic treatment guidance along with nearby dermatologist suggestions.

---

## 🚀 Live Demo

**[Deploy on Hugging Face Spaces](https://huggingface.co/spaces)**

[![Deploy to Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Deploy%20to%20Spaces-blue)](https://huggingface.co/spaces)

---

## 🚀 Features

* 📷 Upload skin image for analysis
* 🤖 Deep learning model for disease prediction
* 💊 Basic treatment recommendations
* 📍 Dermatologist suggestions using Google Places API
* 🌐 Interactive UI built with Streamlit

---

## 🏥 Supported Diseases

* Cellulitis
* Impetigo
* Athlete's Foot
* Nail Fungus
* Ringworm
* Cutaneous Larva Migrans
* Chickenpox
* Shingles

---

## ⚠️ Disclaimer

This application is for educational purposes only and **not a substitute for professional medical advice**. Always consult a qualified doctor for diagnosis and treatment.

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Requests (Google Places API)

---

## 📂 Project Structure

```
Skin Diseases classification/
│
├── skin_disease_app.py        # Main Streamlit app
├── skin_disease_model.h5      # Trained model
├── requirements.txt           # Dependencies
├── .gitignore
├── README.md
│
├── skin_clean/                # Virtual environment (ignored)
├── __pycache__/               # Cache (ignored)
├── temp_image.jpg             # Temporary file (ignored)
```

---

## 🚀 Deployment Options

### Option 1: Hugging Face Spaces (Recommended - Free)

```bash
# 1. Install huggingface-cli
pip install huggingface-hub

# 2. Login
huggingface-cli login

# 3. Create Space
huggingface-cli repo create skin-disease-classifier --type space --space_sdk streamlit

# 4. Push code
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/skin-disease-classifier
git push hf main
```

Then go to your Space → **Settings** → **Repository Secrets** → Add `GOOGLE_API_KEY`

### Option 2: Google Cloud Run (Google Tools Theme)

```bash
# 1. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/skin-disease

# 2. Deploy to Cloud Run
gcloud run deploy skin-disease \
  --image gcr.io/YOUR_PROJECT/skin-disease \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars "GOOGLE_API_KEY=your_key_here"
```

### Option 3: Streamlit Cloud (Simple)

Just connect your GitHub repo on [share.streamlit.io](https://share.streamlit.io). If you get Python 3.14 compatibility errors, update `requirements.txt` to use `tensorflow>=2.20.0` and push.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/<your-username>/skin-disease-classifier.git
cd skin-disease-classifier
```

---

### 2️⃣ Create virtual environment

```
python -m venv skin_env
```

Activate it:

**Windows:**

```
skin_env\Scripts\activate
```

**Mac/Linux:**

```
source skin_env/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Set up API Key (IMPORTANT)

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_api_key_here
```

Or set environment variable:

**Windows:**

```
set GOOGLE_API_KEY=your_api_key_here
```

**Mac/Linux:**

```
export GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ How to Run the Project

```
streamlit run skin_disease_app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

## 🧪 How It Works

1. User uploads an image
2. Image is resized and normalized
3. CNN model predicts disease
4. Result + treatment suggestions displayed
5. Nearby dermatologists fetched via API

---

## 📈 Limitations

* Basic CNN model (not medical-grade accuracy)
* Limited dataset
* No confidence score or explainability
* Depends on internet for doctor recommendations

---

## 🔮 Future Improvements

* Use pretrained models (MobileNet, ResNet)
* Add prediction confidence (%)
* Grad-CAM visualization for explainability
* Deploy on cloud (Streamlit Cloud / AWS)

---

## 👨‍💻 Author

Jashwanth Aravapalli,Sri Sreshta,Gayathri Chinmayiee,Sai Vamshi

---

## ⭐ If you found this useful

Give this repo a star ⭐ and consider improving it further!
