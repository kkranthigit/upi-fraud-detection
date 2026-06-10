import pandas as pd
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