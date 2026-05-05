"""
predict.py — Score individual loan applications
================================================
Usage:
    python predict.py                     # interactive demo
    python predict.py --csv my_apps.csv   # batch scoring
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.data import FEATURES_ALL, FEATURES_NUM, FEATURES_CAT


def load_artifacts():
    try:
        model = joblib.load("models/loan_model.pkl")
        enc   = joblib.load("models/encoders.pkl")
        return model, enc["le_map"], enc["scaler"]
    except FileNotFoundError:
        print("❌  Models not found. Run  train.py  first.")
        sys.exit(1)


def encode_and_scale(df, le_map, scaler):
    df = df.copy()
    for col in FEATURES_CAT:
        df[col] = le_map[col].transform(df[col])
    df[FEATURES_NUM] = scaler.transform(df[FEATURES_NUM])
    return df


def risk_band(prob):
    if prob >= 0.80:   return "🟢 Low Risk"
    elif prob >= 0.55: return "🟡 Medium Risk"
    elif prob >= 0.35: return "🟠 High Risk"
    else:              return "🔴 Very High Risk"


def predict_applications(df_raw, model, le_map, scaler):
    df_enc = encode_and_scale(df_raw[FEATURES_ALL], le_map, scaler)
    probs  = model.predict_proba(df_enc)[:, 1]
    labels = model.predict(df_enc)
    result = df_raw.copy()
    result["approval_probability"] = probs.round(4)
    result["decision"]             = ["✅ APPROVED" if p == 1 else "❌ REJECTED" for p in labels]
    result["risk_band"]            = [risk_band(p) for p in probs]
    return result


# ── Demo applications ──────────────────────────────────────────────────────────
DEMO_APPS = pd.DataFrame([
    # age  income   loan_amt  term   credit  dti    emp_yrs  n_acc  n_derog  education      home          purpose
    [35,   85_000,  25_000,   36,    740,    0.18,  8.0,     6,     0,       "Bachelor",    "Mortgage",   "Home Improvement"],
    [24,   32_000,  15_000,   60,    580,    0.45,  1.5,     3,     2,       "High School", "Rent",       "Debt Consolidation"],
    [52,  145_000,  50_000,   48,    810,    0.12,  20.0,    10,    0,       "Master",      "Own",        "Business"],
    [29,   48_000,  10_000,   24,    620,    0.30,  3.0,     5,     1,       "Associate",   "Rent",       "Medical"],
    [43,   95_000,  35_000,   60,    695,    0.22,  12.0,    7,     0,       "Bachelor",    "Mortgage",   "Education"],
], columns=FEATURES_ALL)


def demo():
    model, le_map, scaler = load_artifacts()
    results = predict_applications(DEMO_APPS, model, le_map, scaler)

    print("\n🏦  Bank Loan Approval System — Demo Predictions")
    print("=" * 65)

    profiles = [
        "Stable homeowner, good credit",
        "Young renter, high DTI, low credit",
        "Senior professional, excellent credit",
        "Mid-range borrower, some derogatory marks",
        "Established borrower, moderate profile",
    ]

    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\n  Applicant #{i+1}: {profiles[i]}")
        print(f"    Income        : ${row['income']:>10,.0f}")
        print(f"    Loan Amount   : ${row['loan_amount']:>10,.0f}")
        print(f"    Credit Score  : {row['credit_score']}")
        print(f"    DTI Ratio     : {row['debt_to_income']:.0%}")
        print(f"    Decision      : {row['decision']}")
        print(f"    Approval Prob : {row['approval_probability']*100:.1f}%")
        print(f"    Risk Band     : {row['risk_band']}")
    print()


def from_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES_ALL if c not in df.columns]
    if missing:
        print(f"❌  Missing columns: {missing}")
        sys.exit(1)
    model, le_map, scaler = load_artifacts()
    results = predict_applications(df, model, le_map, scaler)
    out = path.replace(".csv", "_decisions.csv")
    results.to_csv(out, index=False)
    print(f"✅  Decisions saved to: {out}")
    print(results[["loan_amount", "credit_score", "approval_probability",
                   "decision", "risk_band"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Approval — Inference")
    parser.add_argument("--csv", default=None,
                        help="CSV file with loan applications to score")
    args = parser.parse_args()
    from_csv(args.csv) if args.csv else demo()
