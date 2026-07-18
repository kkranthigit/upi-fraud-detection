import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import joblib
from preprocess import load_and_preprocess

def train():
    X,y = load_and_preprocess('data/transactions.csv')


    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    print("Applying SMOTE...")
    smote= SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train,y_train)
    print(f"After SMOTE: {y_train_sm.value_counts().to_dict()}")

    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )

        print("Training XGBoost...")
        model.fit(X_train_sm,y_train_sm)

        y_pred = model.predict(X_test)
        report = classification_report(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)

        print(report)
        print(f"ROC-AUC Score: {auc:.4f}")


        mlflow.log_param("n_estimators",100)
        mlflow.log_param("max_depth",6)
        mlflow.log_metric("roc_auc",auc)
        mlflow.xgboost.log_model(model,"model")

        joblib.dump(model,'models/fraud_model.pkl')
        print("Model saved to models/fraud_model.pkl")


if __name__ == "__main__":
    train()
