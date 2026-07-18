# UPI Fraud Detection System

A production-grade machine learning system to detect fraudulent UPI transactions in real-time.

## Problem Statement
UPI fraud is a growing concern in India. This system uses XGBoost to classify transactions as fraudulent or legitimate with 99.49% ROC-AUC score.

## Tech Stack
- **ML Model**: XGBoost with SMOTE for class imbalance
- **API**: FastAPI
- **Experiment Tracking**: MLflow
- **Containerization**: Docker
- **Language**: Python 3.10

## Project Structure

upi-fraud-detection/
├── data/                  → dataset
├── src/
│   ├── preprocess.py      → data preprocessing
│   └── train.py           → model training
├── app/
│   └── main.py            → FastAPI endpoints
├── models/                → saved model
├── Dockerfile
└── requirements.txt

## Model Performance
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9949 |
| Recall (Fraud) | 1.00 |
| Accuracy | 99% |

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Health check |
| /predict | POST | Predict fraud |
| /health | GET | API status |

## How to Run

### Local
```bash
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --reload
```

### Docker
```bash
docker build -t upi-fraud-detection .
docker run -p 8000:8000 upi-fraud-detection
```

## API Usage
```json
POST /predict
{
  "step": 1,
  "type": 4,
  "amount": 100000.00,
  "oldbalanceOrg": 100000.00,
  "newbalanceOrig": 0.00,
  "oldbalanceDest": 0.00,
  "newbalanceDest": 100000.00
}
```

## Dataset
- Source: Kaggle — Online Payments Fraud Detection
- Size: 6.3 million transactions
- Fraud rate: 0.13%