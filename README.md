# 🏦 Bank Loan Approval System

An end-to-end **Credit Risk Analysis** pipeline that predicts whether a bank should approve a loan application using **Logistic Regression**, **Random Forest**, and **Gradient Boosting** — with a fully interactive terminal dashboard.

---

## 📌 Problem Statement

Banks receive thousands of loan applications daily. Manually reviewing each one is slow and inconsistent. This project builds an **AI-powered loan approval system** that:

- Predicts approval probability for each applicant
- Assigns a **risk band** (Low / Medium / High / Very High)
- Compares three ML models and picks the best
- Lets officers test applications interactively via a terminal dashboard

---

## 🗂️ Project Structure

```
loan-approval/
│
├── train.py          # Full ML pipeline — generate → preprocess → train → evaluate → save
├── predict.py        # Batch or demo scoring of applications
├── dashboard.py      # 🌟 Interactive terminal dashboard
├── eda.py            # Exploratory Data Analysis
│
├── src/
│   ├── data.py       # Dataset generation + preprocessing utilities
│   └── models.py     # Model training, evaluation, and plotting
│
├── data/
│   └── loans.csv               # Generated dataset (12,000 loan applications)
│
├── models/
│   ├── loan_model.pkl           # Best trained model
│   └── encoders.pkl             # LabelEncoders + StandardScaler
│
├── outputs/
│   ├── evaluation.png           # 2×3 model evaluation dashboard
│   ├── eda_overview.png         # 8-panel EDA overview
│   ├── eda_correlation.png      # Feature correlation heatmap
│   ├── eda_scatter.png          # Credit score vs income scatter
│   └── session_<timestamp>.csv  # Exported dashboard sessions
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/loan-approval-system.git
cd loan-approval-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the Models
```bash
python train.py
```
Generates a 12,000-row synthetic dataset, trains three models, prints metrics, saves the best model, and outputs `outputs/evaluation.png`.

### Step 2 — Explore the Data
```bash
python eda.py
```
Produces 3 EDA chart files in `outputs/`.

### Step 3 — Interactive Dashboard
```bash
python dashboard.py
```
Opens an interactive terminal where you can:
- Enter applicant details (age, income, credit score, DTI, etc.)
- Instantly receive an **Approved / Rejected** decision with probability
- See a **visual probability bar** and risk band
- Track all decisions in the session
- Export the session log to CSV

### Step 4 — Batch Predictions
```bash
python predict.py                      # Demo with 5 sample applicants
python predict.py --csv my_apps.csv    # Score from a CSV file
```

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `age` | Applicant age |
| `income` | Annual income ($) |
| `loan_amount` | Requested loan amount ($) |
| `loan_term` | Repayment period in months |
| `credit_score` | FICO credit score (300–850) |
| `debt_to_income` | Total debt payments / gross income |
| `employment_years` | Years at current/recent employer |
| `num_open_accounts` | Number of open credit accounts |
| `num_derogatory_marks` | Late payments, defaults, bankruptcies |
| `education` | Highest education level attained |
| `home_ownership` | Rent / Own / Mortgage |
| `loan_purpose` | Purpose of the loan |

---

## 🧠 ML Techniques

| Technique | Why |
|---|---|
| **LabelEncoder** | Converts categorical text to numeric |
| **StandardScaler** | Normalizes numeric features |
| **Logistic Regression** | Fast, interpretable baseline |
| **Random Forest** | Handles non-linearity and interactions |
| **Gradient Boosting** | Often best performance on tabular data |
| **ROC-AUC + F1** | Correct metrics for imbalanced classification |

---

## 📈 Sample Results

```
Model                 F1      ROC-AUC   Precision   Recall
──────────────────────────────────────────────────────────
Logistic Regression   0.82    0.90      0.81        0.83
Random Forest         0.87    0.94      0.86        0.88
Gradient Boosting     0.88    0.95      0.87        0.89   ← Best
```

---

## 🖥️ Dashboard Preview

```
╔══════════════════════════════════════════════════════════╗
║       🏦  BANK LOAN APPROVAL SYSTEM  v1.0               ║
║          AI-Powered Credit Risk Analysis                 ║
╚══════════════════════════════════════════════════════════╝

  Decision         :  ✅  APPROVED
  Approval Prob    :  82.3%
  Risk Assessment  :  🟢 LOW RISK
  Probability Bar  :  |████████████████████████░░░░░░|
```

---

## 🔄 Risk Bands

| Band | Approval Probability | Meaning |
|---|---|---|
| 🟢 Low Risk | ≥ 80% | Strong applicant, likely approve |
| 🟡 Medium Risk | 55–79% | Borderline, review manually |
| 🟠 High Risk | 35–54% | Likely reject, high default risk |
| 🔴 Very High Risk | < 35% | Reject, very likely to default |

---

## 🛠️ Future Improvements

- [ ] Add XGBoost / LightGBM for better performance
- [ ] SHAP values for explainable AI (why was this rejected?)
- [ ] Streamlit web dashboard with charts
- [ ] REST API with FastAPI for real-time scoring
- [ ] Fairness analysis (bias across demographic groups)
- [ ] Hyperparameter tuning with Optuna

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

Built as a portfolio project for open-source contributions.  
Contributions and PRs welcome!
