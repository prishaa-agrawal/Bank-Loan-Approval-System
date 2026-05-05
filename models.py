"""
src/models.py — Train, evaluate, and persist ML models
"""

import numpy as np
import pandas as pd
import joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", C=0.5
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        random_state=42
    ),
}


def train_all(X_train, y_train):
    trained = {}
    for name, model in MODELS.items():
        print(f"    ⚙  Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_all(trained, X_test, y_test, feature_names):
    results = {}
    best_name, best_auc = "", 0.0

    for name, model in trained.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "f1":        round(f1_score(y_test, y_pred), 4),
            "auc":       round(roc_auc_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "report":    classification_report(y_test, y_pred,
                             target_names=["Rejected", "Approved"]),
            "cm":        confusion_matrix(y_test, y_pred),
            "y_prob":    y_prob,
        }
        results[name] = metrics

        print(f"\n  ── {name} ──────────────────────────────")
        print(f"     F1={metrics['f1']}  AUC={metrics['auc']}"
              f"  Precision={metrics['precision']}  Recall={metrics['recall']}")
        print(metrics["report"])

        if metrics["auc"] > best_auc:
            best_auc, best_name = metrics["auc"], name

    print(f"\n  🏆  Best model: {best_name}  (ROC-AUC = {best_auc})")
    return results, best_name


def save_best(trained, best_name, path="models/loan_model.pkl"):
    joblib.dump(trained[best_name], path)
    print(f"  💾  Saved → {path}")


def plot_evaluation(trained, results, X_test, y_test, feature_names):
    """Generate a 2×3 evaluation dashboard saved as outputs/evaluation.png"""
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Bank Loan Approval — Model Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    colors = {"Logistic Regression": "#5C85D6",
              "Random Forest":       "#4CAF50",
              "Gradient Boosting":   "#FF7043"}

    # ── ROC Curves ──────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
        ax0.plot(fpr, tpr, label=f"{name} (AUC={metrics['auc']})",
                 color=colors[name], linewidth=2)
    ax0.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax0.set_title("ROC Curves")
    ax0.set_xlabel("False Positive Rate")
    ax0.set_ylabel("True Positive Rate")
    ax0.legend(fontsize=8)

    # ── Precision-Recall Curves ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    for name, metrics in results.items():
        prec, rec, _ = precision_recall_curve(y_test, metrics["y_prob"])
        ax1.plot(rec, prec, label=name, color=colors[name], linewidth=2)
    ax1.set_title("Precision-Recall Curves")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend(fontsize=8)

    # ── Metric Comparison Bar Chart ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    metric_names = ["f1", "auc", "precision", "recall"]
    x = np.arange(len(metric_names))
    width = 0.25
    for i, (name, metrics) in enumerate(results.items()):
        vals = [metrics[m] for m in metric_names]
        bars = ax2.bar(x + i * width, vals, width, label=name,
                       color=colors[name], alpha=0.85)
    ax2.set_title("Metric Comparison")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(["F1", "AUC", "Precision", "Recall"])
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    # ── Confusion Matrices (best model) ─────────────────────────────────────
    best_name_local = max(results, key=lambda k: results[k]["auc"])
    cm = results[best_name_local]["cm"]
    for idx, (row_title, cm_data) in enumerate([
        (f"{best_name_local}\n(Test Set)", cm)
    ]):
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(cm_data, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Rejected", "Approved"])
        ax.set_yticklabels(["Rejected", "Approved"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix\n{best_name_local}")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm_data[r, c]), ha="center",
                        va="center", fontsize=14, fontweight="bold",
                        color="white" if cm_data[r, c] > cm_data.max() / 2 else "black")

    # ── Feature Importances (best tree model) ───────────────────────────────
    tree_models = {k: v for k, v in trained.items()
                   if hasattr(v, "feature_importances_")}
    if tree_models:
        best_tree = max(tree_models,
                        key=lambda k: results[k]["auc"])
        importances = trained[best_tree].feature_importances_
        indices = np.argsort(importances)[-10:]
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.barh(range(len(indices)),
                 importances[indices],
                 color=colors[best_tree], alpha=0.85)
        ax3.set_yticks(range(len(indices)))
        ax3.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        ax3.set_title(f"Top-10 Feature Importances\n({best_tree})")
        ax3.set_xlabel("Importance")
        ax3.grid(axis="x", alpha=0.3)

    # ── Approval Rate by Credit Score bucket ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    # Rebuild raw predictions on test set
    best_model = trained[best_name_local]
    y_pred = best_model.predict(X_test)
    # We need credit score from test — store it externally; skip if not available
    ax4.bar(["Rejected", "Approved"],
            [int((y_pred == 0).sum()), int((y_pred == 1).sum())],
            color=["#EF5350", "#66BB6A"], edgecolor="white", linewidth=1.5)
    ax4.set_title("Predicted Class Distribution\n(Test Set)")
    ax4.set_ylabel("Count")
    for i, v in enumerate([(y_pred == 0).sum(), (y_pred == 1).sum()]):
        ax4.text(i, v + 5, f"{v:,}", ha="center", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    plt.savefig("outputs/evaluation.png", dpi=150, bbox_inches="tight")
    print("  📊  Saved → outputs/evaluation.png")
    plt.close()
