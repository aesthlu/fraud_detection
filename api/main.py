from fastapi import FastAPI
import joblib
import pandas as pd
from src.ingestion import load_data
from src.preprocessing import preprocess
from src.features import build_features

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/fraud_model.pkl")

df = load_data("data/creditcard.csv")
df, scaler = preprocess(df)
df = build_features(df)
FEATURES = df.drop("Class", axis=1).columns.tolist()

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    proba = model.predict_proba(df[FEATURES])[0, 1]

    return {
        "fraud_probability": float(proba),
        "is_fraud": int(proba > 0.2)
    }

# uvicorn api.main:app --reload