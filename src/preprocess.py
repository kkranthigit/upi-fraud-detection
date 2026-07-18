import pandas as pd
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
