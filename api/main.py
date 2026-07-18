from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import preprocess

app = FastAPI(title="UPI Fraud Detection API", version="1.0")

# Load model and preprocessor at startup
model = joblib.load("models/fraud_model.pkl")
artifacts = joblib.load("models/preprocessor.pkl")

class Transaction(BaseModel):
    amount: float
    hour: int
    day_of_week: int
    merchant_category: str
    device_type: str
    is_new_device: int
    failed_attempts: int
    location_mismatch: int

@app.get("/")
def root():
    return {"message": "UPI Fraud Detection API is running 🚀"}

@app.post("/predict")
def predict(txn: Transaction):
    df = pd.DataFrame([txn.model_dump()])

    X, _ = preprocess(df, fit=False, artifacts=artifacts)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW"
    }