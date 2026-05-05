"""
src/data.py — Synthetic loan dataset generator + preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib, os

FEATURES_NUM = [
    "age", "income", "loan_amount", "loan_term",
    "credit_score", "debt_to_income", "employment_years",
    "num_open_accounts", "num_derogatory_marks",
]
FEATURES_CAT = ["education", "home_ownership", "loan_purpose"]
FEATURES_ALL = FEATURES_NUM + FEATURES_CAT
TARGET       = "loan_approved"


def generate_dataset(n=12000, random_state=42):
    np.random.seed(random_state)

    age               = np.random.randint(21, 70, n)
    income            = np.random.lognormal(10.8, 0.6, n).clip(20_000, 500_000).round(-2)
    credit_score      = np.random.normal(670, 80, n).clip(300, 850).round().astype(int)
    loan_amount       = np.random.lognormal(10.2, 0.7, n).clip(1_000, 100_000).round(-2)
    loan_term         = np.random.choice([12, 24, 36, 48, 60, 84], n)
    debt_to_income    = np.random.beta(2, 5, n).round(3)          # 0–1
    employment_years  = np.random.exponential(5, n).clip(0, 40).round(1)
    num_open_accounts = np.random.poisson(5, n).clip(0, 20)
    num_derogatory    = np.random.poisson(0.4, n).clip(0, 8)
    education         = np.random.choice(
        ["High School", "Associate", "Bachelor", "Master", "PhD"],
        n, p=[0.25, 0.15, 0.35, 0.18, 0.07]
    )
    home_ownership    = np.random.choice(
        ["Rent", "Own", "Mortgage"], n, p=[0.35, 0.25, 0.40]
    )
    loan_purpose      = np.random.choice(
        ["Debt Consolidation", "Home Improvement", "Business",
         "Medical", "Education", "Other"],
        n, p=[0.35, 0.20, 0.15, 0.10, 0.12, 0.08]
    )

    # Approval logic — realistic scoring
    score = (
        (credit_score - 300) / 550 * 40          # 0-40 pts
        + (1 - debt_to_income)         * 25      # 0-25 pts
        + np.log1p(income) / np.log1p(500_000) * 20  # 0-20 pts
        + employment_years / 40        * 10      # 0-10 pts
        - num_derogatory               *  3      # penalty
        + np.random.normal(0, 3, n)              # noise
    )
    prob_approval = 1 / (1 + np.exp(-(score - 50) / 8))
    approved = (np.random.uniform(0, 1, n) < prob_approval).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income, "loan_amount": loan_amount,
        "loan_term": loan_term, "credit_score": credit_score,
        "debt_to_income": debt_to_income, "employment_years": employment_years,
        "num_open_accounts": num_open_accounts, "num_derogatory_marks": num_derogatory,
        "education": education, "home_ownership": home_ownership,
        "loan_purpose": loan_purpose, TARGET: approved,
    })
    return df


def preprocess(df, fit=True, encoders_path="models/encoders.pkl"):
    """Encode categoricals + scale numerics. fit=True during training."""
    df = df.copy()

    if fit:
        le_map = {}
        for col in FEATURES_CAT:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_map[col] = le
        scaler = StandardScaler()
        df[FEATURES_NUM] = scaler.fit_transform(df[FEATURES_NUM])
        os.makedirs("models", exist_ok=True)
        joblib.dump({"le_map": le_map, "scaler": scaler}, encoders_path)
    else:
        enc = joblib.load(encoders_path)
        le_map, scaler = enc["le_map"], enc["scaler"]
        for col in FEATURES_CAT:
            df[col] = le_map[col].transform(df[col])
        df[FEATURES_NUM] = scaler.transform(df[FEATURES_NUM])

    return df


def get_splits(df, test_size=0.2, random_state=42):
    X = df[FEATURES_ALL]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)
