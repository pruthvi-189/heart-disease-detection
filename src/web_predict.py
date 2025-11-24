import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "heart_model.joblib")

def load_model():
    return joblib.load(MODEL_PATH)

def predict_sample(model, input_dict):
    df = pd.DataFrame([input_dict])
    probability = model.predict_proba(df)[0][1]
    prediction = int(model.predict(df)[0])

    return {
        "prediction": prediction,
        "probability": float(probability)
    }
