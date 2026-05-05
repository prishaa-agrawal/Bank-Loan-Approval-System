"""
train.py — Full Bank Loan Approval training pipeline
=====================================================
Steps:
  1. Generate / load dataset
  2. Preprocess (encode + scale)
  3. Train Logistic Regression, Random Forest, Gradient Boosting
  4. Evaluate all three and pick best by ROC-AUC
  5. Save best model + encoders
  6. Plot evaluation dashboard → outputs/evaluation.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.data   import generate_dataset, preprocess, get_splits, FEATURES_ALL
from src.models import train_all, evaluate_all, save_best, plot_evaluation
import warnings
warnings.filterwarnings("ignore")


def main():
    print("\n🏦  Bank Loan Approval System — Training Pipeline")
    print("=" * 55)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1/5] Generating dataset …")
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n=12000)
    df.to_csv("data/loans.csv", index=False)
    n_approved = df["loan_approved"].sum()
    n_total    = len(df)
    print(f"      {n_total:,} rows  |  Approved: {n_approved:,} ({n_approved/n_total*100:.1f}%)  "
          f"|  Rejected: {n_total - n_approved:,}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing (encode + scale) …")
    df_proc = preprocess(df, fit=True)

    # ── 3. Split ──────────────────────────────────────────────────────────────
    print("\n[3/5] Train / test split (80 / 20, stratified) …")
    X_train, X_test, y_train, y_test = get_splits(df_proc)
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[4/5] Training models …")
    trained = train_all(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating …")
    results, best_name = evaluate_all(trained, X_test, y_test, FEATURES_ALL)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_best(trained, best_name)
    plot_evaluation(trained, results, X_test, y_test, FEATURES_ALL)

    print("\n✅  Training complete!")
    print("    ▸ Run  python predict.py          for a quick demo")
    print("    ▸ Run  python dashboard.py        for the interactive dashboard")
    print("    ▸ Run  python eda.py              for EDA charts\n")


if __name__ == "__main__":
    main()
