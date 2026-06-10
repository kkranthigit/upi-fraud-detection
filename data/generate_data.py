import pandas as pd
import numpy as np
import random
import os

def generate_upi_data(n_transactions=50000, fraud_rate=0.02, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    def make_transactions(n, is_fraud):
        return {
            "amount": (
                np.random.exponential(scale=8000, size=n).clip(10, 100000)
                if is_fraud else
                np.random.exponential(scale=1500, size=n).clip(10, 50000)
            ),
            "hour": (
                np.random.choice(range(0, 5), size=n)
                if is_fraud else
                np.random.choice(range(8, 22), size=n)
            ),
            "day_of_week": np.random.randint(0, 7, size=n),
            "merchant_category": np.random.choice(
                ["groceries", "electronics", "travel", "unknown", "crypto"],
                size=n,
                p=[0.05, 0.05, 0.05, 0.70, 0.15] if is_fraud else [0.35, 0.20, 0.25, 0.10, 0.10]
            ),
            "device_type": np.random.choice(
                ["mobile", "desktop", "unknown"],
                size=n,
                p=[0.20, 0.10, 0.70] if is_fraud else [0.75, 0.20, 0.05]
            ),
            "is_new_device": np.random.choice(
                [0, 1], size=n,
                p=[0.20, 0.80] if is_fraud else [0.90, 0.10]
            ),
            "failed_attempts": (
                np.random.randint(2, 6, size=n)
                if is_fraud else
                np.random.randint(0, 2, size=n)
            ),
            "location_mismatch": np.random.choice(
                [0, 1], size=n,
                p=[0.30, 0.70] if is_fraud else [0.95, 0.05]
            ),
            "is_fraud": int(is_fraud)
        }

    df = pd.concat([
        pd.DataFrame(make_transactions(n_legit, False)),
        pd.DataFrame(make_transactions(n_fraud, True))
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = generate_upi_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"✅ Dataset saved! Total: {len(df)} rows, Fraud: {df['is_fraud'].sum()} transactions")