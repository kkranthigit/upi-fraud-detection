import pandas as pd
<<<<<<< HEAD
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    df=pd.read_csv(filepath)


    le = LabelEncoder()
    df['type']=le.fit_transform(df['type'])

    df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'],axis=1)

    X=df.drop('isFraud',axis=1)
    y=df['isFraud']

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {y.sum()} ({y.mean()*100:.2f}%)")

    return X,y


if __name__ == "__main__":
    X,y = load_and_preprocess('data/transactions.csv')
    print(X.head())
=======
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

CATEGORICAL_COLS = ["merchant_category", "device_type"]
NUMERIC_COLS = ["amount", "hour", "day_of_week", "failed_attempts"]
BINARY_COLS = ["is_new_device", "location_mismatch"]

def preprocess(df: pd.DataFrame, fit=True, artifacts=None):
    df = df.copy()

    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

        artifacts = {"encoders": encoders, "scaler": scaler}
        os.makedirs("models", exist_ok=True)
        joblib.dump(artifacts, "models/preprocessor.pkl")
        print("✅ Preprocessor saved!")

    else:
        for col in CATEGORICAL_COLS:
            df[col] = artifacts["encoders"][col].transform(df[col])

        df[NUMERIC_COLS] = artifacts["scaler"].transform(df[NUMERIC_COLS])

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS
    X = df[feature_cols]
    return X, artifacts
>>>>>>> 06a43c8d25122e07a969b15b2b80416462b5a840
