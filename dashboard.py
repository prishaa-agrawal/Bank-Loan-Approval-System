"""
dashboard.py — Interactive Bank Loan Approval Dashboard
========================================================
A terminal-based interactive dashboard that lets you:
  • Enter loan application details manually
  • See instant approval decision + risk band
  • View running statistics of all predictions made
  • Export session results to CSV

Run:  python dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from src.data import FEATURES_ALL, FEATURES_NUM, FEATURES_CAT

# ── ANSI colors ───────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;214m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}
  ╔══════════════════════════════════════════════════════════╗
  ║       🏦  BANK LOAN APPROVAL SYSTEM  v1.0               ║
  ║          AI-Powered Credit Risk Analysis                 ║
  ╚══════════════════════════════════════════════════════════╝
{C.RESET}""")


def load_artifacts():
    try:
        model = joblib.load("models/loan_model.pkl")
        enc   = joblib.load("models/encoders.pkl")
        print(f"  {C.GREEN}✓{C.RESET}  Model loaded successfully\n")
        return model, enc["le_map"], enc["scaler"]
    except FileNotFoundError:
        print(f"\n  {C.RED}❌  Models not found. Run  train.py  first.{C.RESET}\n")
        sys.exit(1)


def prompt(label, type_fn, valid_range=None, choices=None, default=None):
    """Generic input prompt with validation."""
    while True:
        try:
            hint = ""
            if choices:
                hint = f"\n     Options: {', '.join(choices)}"
            elif valid_range:
                hint = f"  [{valid_range[0]}–{valid_range[1]}]"
            if default is not None:
                hint += f"  (default: {default})"

            raw = input(f"  {C.CYAN}{label}{C.RESET}{hint}: ").strip()
            if raw == "" and default is not None:
                return default
            val = type_fn(raw)
            if choices and str(val) not in choices:
                raise ValueError
            if valid_range and not (valid_range[0] <= val <= valid_range[1]):
                raise ValueError
            return val
        except (ValueError, KeyError):
            err = f"choices: {choices}" if choices else f"range {valid_range}"
            print(f"  {C.RED}  ✗  Invalid input. Expected {err}.{C.RESET}")


def collect_application():
    """Interactive form to collect a loan application."""
    print(f"\n{C.BOLD}{C.WHITE}  ── Applicant Details ──────────────────────────────{C.RESET}")
    age              = prompt("Age",                      int,   (18, 80),        default=35)
    income           = prompt("Annual Income ($)",        float, (10_000, 1_000_000), default=70000)
    credit_score     = prompt("Credit Score",             int,   (300, 850),      default=680)
    employment_years = prompt("Years Employed",           float, (0, 50),         default=5.0)
    num_derogatory   = prompt("Derogatory Marks",         int,   (0, 10),         default=0)
    num_open_accs    = prompt("Open Accounts",            int,   (0, 30),         default=5)

    print(f"\n{C.BOLD}{C.WHITE}  ── Loan Details ────────────────────────────────────{C.RESET}")
    loan_amount  = prompt("Loan Amount ($)",          float, (500, 200_000),   default=20000)
    loan_term    = prompt("Loan Term (months)",       int,   None,
                          choices=["12","24","36","48","60","84"], default=36)
    debt_to_income = prompt("Debt-to-Income Ratio",   float, (0.0, 1.0),      default=0.25)
    loan_purpose = prompt("Loan Purpose",             str,   None,
                          choices=["Debt Consolidation","Home Improvement",
                                   "Business","Medical","Education","Other"],
                          default="Home Improvement")

    print(f"\n{C.BOLD}{C.WHITE}  ── Personal Profile ────────────────────────────────{C.RESET}")
    education      = prompt("Education Level",  str, None,
                            choices=["High School","Associate","Bachelor","Master","PhD"],
                            default="Bachelor")
    home_ownership = prompt("Home Ownership",   str, None,
                            choices=["Rent","Own","Mortgage"],
                            default="Rent")

    return pd.DataFrame([{
        "age": age, "income": income, "loan_amount": loan_amount,
        "loan_term": loan_term, "credit_score": credit_score,
        "debt_to_income": debt_to_income, "employment_years": employment_years,
        "num_open_accounts": num_open_accs, "num_derogatory_marks": num_derogatory,
        "education": education, "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
    }])


def encode_and_score(df_raw, model, le_map, scaler):
    df = df_raw.copy()
    for col in FEATURES_CAT:
        df[col] = le_map[col].transform(df[col])
    df[FEATURES_NUM] = scaler.transform(df[FEATURES_NUM])
    prob  = model.predict_proba(df[FEATURES_ALL])[:, 1][0]
    label = model.predict(df[FEATURES_ALL])[0]
    return prob, label


def risk_band(prob):
    if prob >= 0.80:   return C.GREEN  + "🟢 LOW RISK"       + C.RESET
    elif prob >= 0.55: return C.YELLOW + "🟡 MEDIUM RISK"    + C.RESET
    elif prob >= 0.35: return C.ORANGE + "🟠 HIGH RISK"      + C.RESET
    else:              return C.RED    + "🔴 VERY HIGH RISK" + C.RESET


def display_decision(prob, label, app):
    approved = label == 1
    bar_len  = int(prob * 30)
    bar      = ("█" * bar_len).ljust(30)
    color    = C.GREEN if approved else C.RED
    decision = "✅  APPROVED" if approved else "❌  REJECTED"

    print(f"""
{C.BOLD}  ╔══════════════════════════════════════════════════╗
  ║            LOAN DECISION RESULT                  ║
  ╚══════════════════════════════════════════════════╝{C.RESET}

  Decision         :  {color}{C.BOLD}{decision}{C.RESET}
  Approval Prob    :  {color}{prob*100:.1f}%{C.RESET}
  Risk Assessment  :  {risk_band(prob)}

  Probability Bar  :  {color}|{bar}|{C.RESET}

  Key Factors:
    Credit Score   :  {app['credit_score'].values[0]}
    DTI Ratio      :  {app['debt_to_income'].values[0]:.0%}
    Employment     :  {app['employment_years'].values[0]:.1f} yrs
    Derogatory     :  {app['num_derogatory_marks'].values[0]}
""")


def show_session_stats(history):
    if not history:
        return
    n          = len(history)
    n_approved = sum(1 for h in history if h["approved"])
    avg_prob   = np.mean([h["prob"] for h in history])

    print(f"""
{C.BOLD}{C.CYAN}  ── Session Statistics ({n} applications) ─────────────{C.RESET}
  Approved      : {C.GREEN}{n_approved}{C.RESET} / {n}  ({n_approved/n*100:.1f}%)
  Rejected      : {C.RED}{n - n_approved}{C.RESET} / {n}  ({(n-n_approved)/n*100:.1f}%)
  Avg Approval  : {avg_prob*100:.1f}%
""")


def export_session(history):
    if not history:
        print(f"  {C.GREY}No applications to export.{C.RESET}\n")
        return
    rows = []
    for h in history:
        row = h["app"].copy()
        row["approval_probability"] = round(h["prob"], 4)
        row["decision"]             = "Approved" if h["approved"] else "Rejected"
        row["timestamp"]            = h["ts"]
        rows.append(row)
    df = pd.concat(rows, ignore_index=True)
    fname = f"outputs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(fname, index=False)
    print(f"  {C.GREEN}✅  Exported → {fname}{C.RESET}\n")


def main():
    banner()
    model, le_map, scaler = load_artifacts()
    history = []

    while True:
        print(f"{C.BOLD}  Menu{C.RESET}")
        print(f"  {C.CYAN}[1]{C.RESET} New Loan Application")
        print(f"  {C.CYAN}[2]{C.RESET} View Session Stats")
        print(f"  {C.CYAN}[3]{C.RESET} Export Session to CSV")
        print(f"  {C.CYAN}[4]{C.RESET} Quit\n")

        choice = input(f"  {C.BOLD}Enter choice [1-4]: {C.RESET}").strip()

        if choice == "1":
            app = collect_application()
            prob, label = encode_and_score(app, model, le_map, scaler)
            display_decision(prob, label, app)
            history.append({
                "app": app, "prob": prob,
                "approved": label == 1,
                "ts": datetime.now().isoformat(),
            })

        elif choice == "2":
            show_session_stats(history)

        elif choice == "3":
            export_session(history)

        elif choice == "4":
            print(f"\n  {C.CYAN}Goodbye! 👋{C.RESET}\n")
            break

        else:
            print(f"  {C.RED}Invalid choice. Please enter 1–4.{C.RESET}\n")


if __name__ == "__main__":
    main()
