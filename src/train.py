import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

from src.preprocess import preprocess

def train():
    # 1. Load data
    print("📂 Loading data...")
    df = pd.read_csv("data/transactions.csv")
    y = df["is_fraud"]
    df = df.drop(columns=["is_fraud"])

    # 2. Preprocess
    print("⚙️  Preprocessing...")
    X, artifacts = preprocess(df, fit=True)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 4. Handle class imbalance with SMOTE
    print("⚖️  Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Fraud: {y_train_res.sum()}, Legit: {(y_train_res==0).sum()}")

    # 5. Train XGBoost
    print("🚀 Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # 7. Save model
    joblib.dump(model, "models/fraud_model.pkl")
    print("✅ Model saved to models/fraud_model.pkl")

if __name__ == "__main__":
    train()