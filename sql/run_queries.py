"""Execute every .sql file in this directory against data/churn.db and
print the result tables. Also used by model/train_model.py and the
dashboard build step to pull real numbers.
"""
import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "data" / "churn.db"

QUERY_FILES = [
    "cohort_retention.sql",
    "churn_by_contract_tenure.sql",
    "revenue_at_risk.sql",
]


def run_query_file(filename: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    sql = (HERE / filename).read_text()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def run_all() -> dict:
    return {name: run_query_file(name) for name in QUERY_FILES}


def main():
    results = run_all()
    for name, df in results.items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
