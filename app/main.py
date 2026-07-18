from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="UPI Fraud Detection API")

# load model
model = joblib.load('models/fraud_model.pkl')

class Transaction(BaseModel):
    step: int
    type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.get("/")
def home():
    return {"message": "UPI Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    features = np.array([[
        transaction.step,
        transaction.type,
        transaction.amount,
        transaction.oldbalanceOrg,
        transaction.newbalanceOrig,
        transaction.oldbalanceDest,
        transaction.newbalanceDest
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}