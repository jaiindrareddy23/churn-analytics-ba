"""Compute the real numbers the static dashboard embeds, from the actual
SQL query results and model metrics -- no fabricated figures.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "sql"))
from run_queries import run_all  # noqa: E402

OUT_PATH = HERE / "dashboard_data.json"


def retention_curve(cohort_df):
    """Average retention % at each months-since-signup, weighted by cohort
    size, restricted to cohorts old enough to have reached that horizon."""
    agg = (
        cohort_df.groupby("months_since_signup")
        .apply(
            lambda g: round(100.0 * (g["active_customers"].sum() / g["cohort_size"].sum()), 1)
        )
        .reset_index(name="avg_retention_pct")
    )
    return agg.to_dict(orient="records")


def main():
    results = run_all()
    cohort_df = results["cohort_retention.sql"]
    churn_tenure_df = results["churn_by_contract_tenure.sql"]
    revenue_df = results["revenue_at_risk.sql"]

    model_summary_path = ROOT / "model" / "model_summary.json"
    model_summary = json.loads(model_summary_path.read_text()) if model_summary_path.exists() else {}

    data = {
        "retention_curve": retention_curve(cohort_df),
        "churn_by_contract_tenure": churn_tenure_df.to_dict(orient="records"),
        "revenue_at_risk": revenue_df.to_dict(orient="records"),
        "model_metrics": model_summary.get("metrics", []),
    }
    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(data, indent=2)[:800])


if __name__ == "__main__":
    main()
