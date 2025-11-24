import sys
import os

# ----------------------------------------
# FIX: Add project root directory to path
# ----------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
from src.web_predict import load_model, predict_sample


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Detector", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below to predict the likelihood of heart disease.")


# -----------------------------
# Input Form
# -----------------------------
age = st.number_input("Age", 1, 120, 55)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1], index=1)
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 130)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
exang = st.selectbox("Exercise-Induced Angina (0/1)", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])


# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):
    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    model = load_model()
    output = predict_sample(model, input_data)

    prediction = output["prediction"]
    probability = output["probability"]

    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected\nProbability: **{probability:.2f}**")
    else:
        st.success(f"✅ No Heart Disease Detected\nProbability: **{probability:.2f}**")
