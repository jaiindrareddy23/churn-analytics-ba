"""Generate a synthetic but realistic telecom/subscription churn dataset.

Writes data/customers.csv and loads it (plus a derived monthly_activity
table) into a SQLite database at data/churn.db.
"""
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 5000
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "customers.csv"
DB_PATH = HERE / "churn.db"

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]
SIGNUP_START = date(2022, 1, 1)
SIGNUP_END = date(2025, 10, 1)


def _random_signup_dates(rng, n):
    start_ord = SIGNUP_START.toordinal()
    end_ord = SIGNUP_END.toordinal()
    offsets = rng.integers(start_ord, end_ord, size=n)
    return [date.fromordinal(int(o)) for o in offsets]


def generate(n=N_CUSTOMERS, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{i:06d}" for i in range(1, n + 1)]
    signup_date = _random_signup_dates(rng, n)
    today = date(2025, 11, 1)
    max_tenure = np.array([(today - d).days // 30 for d in signup_date])
    tenure_months = np.minimum(
        rng.poisson(lam=18, size=n).clip(min=0), max_tenure
    ).clip(min=0)

    contract_type = rng.choice(CONTRACT_TYPES, size=n, p=[0.55, 0.27, 0.18])
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=[0.34, 0.19, 0.24, 0.23])
    internet_service = rng.choice(INTERNET_SERVICE, size=n, p=[0.34, 0.44, 0.22])

    base_charge = rng.normal(55, 12, size=n)
    fiber_bonus = np.where(internet_service == "Fiber optic", 22, 0)
    dsl_bonus = np.where(internet_service == "DSL", 8, 0)
    monthly_charges = (base_charge + fiber_bonus + dsl_bonus).clip(18, 140).round(2)

    total_charges = (monthly_charges * tenure_months * rng.uniform(0.92, 1.05, size=n)).round(2)

    avg_monthly_usage_gb = (
        rng.gamma(shape=4.0, scale=np.where(internet_service == "Fiber optic", 18, 9), size=n)
    ).round(1)

    support_tickets = rng.poisson(lam=1.3, size=n)
    support_tickets = np.where(rng.random(n) < 0.12, support_tickets + rng.poisson(3, n), support_tickets)

    num_dependents = rng.poisson(lam=0.6, size=n).clip(0, 5)
    senior_citizen = (rng.random(n) < 0.16).astype(int)
    paperless_billing = (rng.random(n) < 0.59).astype(int)
    autopay = (rng.random(n) < 0.46).astype(int)

    satisfaction_score = (
        7.2
        - 0.35 * support_tickets
        - np.where(contract_type == "Month-to-month", 0.6, 0.0)
        + rng.normal(0, 1.1, size=n)
    ).clip(1, 10).round(1)

    # Latent churn propensity: believable, noisy, not perfectly separable.
    contract_risk = np.select(
        [contract_type == "Month-to-month", contract_type == "One year", contract_type == "Two year"],
        [1.3, 0.1, -0.9],
    )
    payment_risk = np.where(payment_method == "Electronic check", 0.55, -0.1)
    tenure_term = -0.055 * tenure_months
    charge_term = 0.012 * (monthly_charges - 60)
    ticket_term = 0.28 * support_tickets
    satisfaction_term = -0.33 * (satisfaction_score - 5.5)
    fiber_term = np.where(internet_service == "Fiber optic", 0.25, 0.0)
    noise = rng.normal(0, 0.85, size=n)

    logit = (
        -1.35
        + contract_risk
        + payment_risk
        + tenure_term
        + charge_term
        + ticket_term
        + satisfaction_term
        + fiber_term
        + noise
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "signup_date": [d.isoformat() for d in signup_date],
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "support_tickets": support_tickets,
            "num_dependents": num_dependents,
            "senior_citizen": senior_citizen,
            "paperless_billing": paperless_billing,
            "autopay": autopay,
            "satisfaction_score": satisfaction_score,
            "churn": churn,
        }
    )
    return df


def build_monthly_activity(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Derived per-customer-month activity rows for cohort analysis."""
    rows = []
    for cid, signup_str, tenure, churned, mcharge in zip(
        df["customer_id"], df["signup_date"], df["tenure_months"], df["churn"], df["monthly_charges"]
    ):
        signup = date.fromisoformat(signup_str)
        active_months = int(tenure) + (0 if churned else 1)
        active_months = max(active_months, 1)
        for m in range(active_months):
            activity_month = (signup.replace(day=1) + timedelta(days=31 * m)).replace(day=1)
            is_last = (m == active_months - 1) and bool(churned)
            usage = max(5.0, rng.normal(40, 15))
            rows.append(
                (
                    cid,
                    activity_month.isoformat(),
                    m,
                    round(float(mcharge) * float(rng.uniform(0.95, 1.03)), 2),
                    round(usage, 1),
                    int(is_last),
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "activity_month",
            "months_since_signup",
            "charge_amount",
            "usage_gb",
            "is_churn_month",
        ],
    )


def load_into_sqlite(df: pd.DataFrame, activity_df: pd.DataFrame, db_path: Path):
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY,
                signup_date TEXT NOT NULL,
                tenure_months INTEGER NOT NULL,
                contract_type TEXT NOT NULL,
                payment_method TEXT NOT NULL,
                internet_service TEXT NOT NULL,
                monthly_charges REAL NOT NULL,
                total_charges REAL NOT NULL,
                avg_monthly_usage_gb REAL NOT NULL,
                support_tickets INTEGER NOT NULL,
                num_dependents INTEGER NOT NULL,
                senior_citizen INTEGER NOT NULL,
                paperless_billing INTEGER NOT NULL,
                autopay INTEGER NOT NULL,
                satisfaction_score REAL NOT NULL,
                churn INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE monthly_activity (
                customer_id TEXT NOT NULL,
                activity_month TEXT NOT NULL,
                months_since_signup INTEGER NOT NULL,
                charge_amount REAL NOT NULL,
                usage_gb REAL NOT NULL,
                is_churn_month INTEGER NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
            """
        )
        df.to_sql("customers", conn, if_exists="append", index=False)
        activity_df.to_sql("monthly_activity", conn, if_exists="append", index=False)
        conn.execute("CREATE INDEX idx_activity_customer ON monthly_activity (customer_id)")
        conn.execute("CREATE INDEX idx_activity_month ON monthly_activity (activity_month)")
        conn.execute("CREATE INDEX idx_customers_contract ON customers (contract_type)")
        conn.commit()
    finally:
        conn.close()


def main():
    rng = np.random.default_rng(SEED)
    df = generate()
    df.to_csv(CSV_PATH, index=False)
    activity_df = build_monthly_activity(df, rng)
    load_into_sqlite(df, activity_df, DB_PATH)

    print(f"Wrote {len(df)} customers to {CSV_PATH}")
    print(f"Wrote {len(activity_df)} activity rows to {DB_PATH} (monthly_activity)")
    print(f"Overall churn rate: {df['churn'].mean():.3%}")


if __name__ == "__main__":
    main()
