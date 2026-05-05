"""
eda.py — Exploratory Data Analysis for the Loan Approval dataset.
Run AFTER train.py (which creates data/loans.csv).
Outputs 4 charts into outputs/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, sys
warnings.import = None

import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ── Load data ────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("data/loans.csv")
except FileNotFoundError:
    print("❌  data/loans.csv not found. Run train.py first.")
    sys.exit(1)

approved   = df[df["loan_approved"] == 1]
rejected   = df[df["loan_approved"] == 0]
COLORS     = {1: "#4CAF50", 0: "#EF5350"}
LABEL_MAP  = {1: "Approved", 0: "Rejected"}
print(f"Dataset: {len(df):,} rows | Approved: {len(approved):,} | Rejected: {len(rejected):,}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Overview dashboard
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Bank Loan Approval — EDA Overview", fontsize=16, fontweight="bold")
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# 1a Approval rate pie
ax = fig.add_subplot(gs[0, 0])
counts = df["loan_approved"].value_counts()
ax.pie(counts.values, labels=["Approved", "Rejected"],
       colors=["#4CAF50", "#EF5350"], autopct="%1.1f%%",
       startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Overall Approval Rate")

# 1b Credit score distribution
ax = fig.add_subplot(gs[0, 1])
for cls in [1, 0]:
    ax.hist(df[df["loan_approved"] == cls]["credit_score"],
            bins=40, alpha=0.65, color=COLORS[cls], label=LABEL_MAP[cls])
ax.axvline(df["credit_score"].median(), color="navy", linestyle="--",
           linewidth=1.2, label="Median")
ax.set_title("Credit Score Distribution")
ax.set_xlabel("Credit Score")
ax.legend(fontsize=8)

# 1c Income distribution (log)
ax = fig.add_subplot(gs[0, 2])
for cls in [1, 0]:
    ax.hist(np.log10(df[df["loan_approved"] == cls]["income"] + 1),
            bins=40, alpha=0.65, color=COLORS[cls], label=LABEL_MAP[cls])
ax.set_title("Income Distribution (log₁₀)")
ax.set_xlabel("log₁₀(Income)")
ax.legend(fontsize=8)

# 1d DTI ratio
ax = fig.add_subplot(gs[0, 3])
for cls in [1, 0]:
    ax.hist(df[df["loan_approved"] == cls]["debt_to_income"],
            bins=40, alpha=0.65, color=COLORS[cls], label=LABEL_MAP[cls])
ax.set_title("Debt-to-Income Ratio")
ax.set_xlabel("DTI Ratio")
ax.legend(fontsize=8)

# 1e Approval by home ownership
ax = fig.add_subplot(gs[1, 0])
grp = df.groupby("home_ownership")["loan_approved"].mean().sort_values()
grp.plot(kind="barh", ax=ax, color="#5C85D6", edgecolor="white")
ax.set_title("Approval Rate by\nHome Ownership")
ax.set_xlabel("Approval Rate")
for i, v in enumerate(grp.values):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
ax.set_xlim(0, 1)

# 1f Approval by education
ax = fig.add_subplot(gs[1, 1])
grp = df.groupby("education")["loan_approved"].mean().sort_values()
grp.plot(kind="barh", ax=ax, color="#FF9800", edgecolor="white")
ax.set_title("Approval Rate by\nEducation Level")
ax.set_xlabel("Approval Rate")
for i, v in enumerate(grp.values):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
ax.set_xlim(0, 1)

# 1g Approval by loan purpose
ax = fig.add_subplot(gs[1, 2])
grp = df.groupby("loan_purpose")["loan_approved"].mean().sort_values()
grp.plot(kind="barh", ax=ax, color="#9C27B0", edgecolor="white")
ax.set_title("Approval Rate by\nLoan Purpose")
ax.set_xlabel("Approval Rate")
for i, v in enumerate(grp.values):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
ax.set_xlim(0, 1)

# 1h Employment years vs approval
ax = fig.add_subplot(gs[1, 3])
for cls in [1, 0]:
    ax.hist(df[df["loan_approved"] == cls]["employment_years"],
            bins=30, alpha=0.65, color=COLORS[cls], label=LABEL_MAP[cls])
ax.set_title("Employment Years")
ax.set_xlabel("Years Employed")
ax.legend(fontsize=8)

plt.savefig("outputs/eda_overview.png", dpi=150, bbox_inches="tight")
print("✅  Saved: outputs/eda_overview.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Correlation heatmap
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 9))
num_cols = ["age", "income", "loan_amount", "loan_term", "credit_score",
            "debt_to_income", "employment_years", "num_open_accounts",
            "num_derogatory_marks", "loan_approved"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_correlation.png", dpi=150, bbox_inches="tight")
print("✅  Saved: outputs/eda_correlation.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Credit score vs income scatter (sample 2k pts)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Credit Score vs Income — by Approval Status",
             fontsize=14, fontweight="bold")

sample = df.sample(2000, random_state=42)
for cls in [0, 1]:
    sub = sample[sample["loan_approved"] == cls]
    axes[0].scatter(sub["credit_score"], np.log10(sub["income"] + 1),
                    alpha=0.4, s=18, color=COLORS[cls], label=LABEL_MAP[cls])
axes[0].set_xlabel("Credit Score")
axes[0].set_ylabel("log₁₀(Income)")
axes[0].set_title("All Purposes")
axes[0].legend()

for cls in [0, 1]:
    sub = sample[sample["loan_approved"] == cls]
    axes[1].scatter(sub["debt_to_income"], sub["credit_score"],
                    alpha=0.4, s=18, color=COLORS[cls], label=LABEL_MAP[cls])
axes[1].set_xlabel("Debt-to-Income Ratio")
axes[1].set_ylabel("Credit Score")
axes[1].set_title("DTI vs Credit Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/eda_scatter.png", dpi=150, bbox_inches="tight")
print("✅  Saved: outputs/eda_scatter.png")
plt.close()

print("\n📊  EDA complete! Charts saved to outputs/\n")
