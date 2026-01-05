from fastapi import FastAPI
import joblib
import pandas as pd
from src.preprocessing import preprocess_inference
from src.features import build_features

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/amount_scaler.pkl")
FEATURES = joblib.load("models/features.pkl")
threshold = joblib.load("models/optimal_threshold.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])

    df = preprocess_inference(df, scaler)
    df = build_features(df)

    proba = model.predict_proba(df[FEATURES])[0, 1]

    return {
        "fraud_probability": float(proba),
        "is_fraud": int(proba > threshold)
    }


# uvicorn api.main:app --reload