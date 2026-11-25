"""Train and compare two churn models (logistic regression, gradient
boosting) on the synthetic customer dataset, saving metrics, a
feature-importance table, and ROC / confusion-matrix plots.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "customers.csv"
RANDOM_STATE = 42

CATEGORICAL = ["contract_type", "payment_method", "internet_service"]
NUMERIC = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_monthly_usage_gb",
    "support_tickets",
    "num_dependents",
    "senior_citizen",
    "paperless_billing",
    "autopay",
    "satisfaction_score",
]


def load_features():
    df = pd.read_csv(DATA_PATH)
    y = df["churn"]
    x = pd.get_dummies(df[CATEGORICAL + NUMERIC], columns=CATEGORICAL, drop_first=True)
    return x, y


def evaluate(name, model, x_test, y_test):
    proba = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }, proba, pred


def main():
    x, y = load_features()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(x_train_scaled, y_train)

    gbc = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gbc.fit(x_train, y_train)

    metrics = []
    m1, proba_lr, _ = evaluate("Logistic Regression", logreg, x_test_scaled, y_test)
    m2, proba_gb, _ = evaluate("Gradient Boosting", gbc, x_test, y_test)
    metrics.extend([m1, m2])
    metrics_df = pd.DataFrame(metrics)

    importance_df = (
        pd.DataFrame({"feature": x.columns, "importance": gbc.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["importance"] = importance_df["importance"].round(4)

    coef_df = (
        pd.DataFrame({"feature": x.columns, "coefficient": logreg.coef_[0]})
        .sort_values("coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )
    coef_df["coefficient"] = coef_df["coefficient"].round(4)

    # ROC curve plot (both models).
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba_lr, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, proba_gb, name="Gradient Boosting", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_title("ROC Curve: Churn Prediction Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix for the better model (gradient boosting).
    pred_gb = gbc.predict(x_test)
    cm = confusion_matrix(y_test, pred_gb)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix: Gradient Boosting")
    fig.tight_layout()
    fig.savefig(HERE / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    metrics_df.to_csv(HERE / "metrics.csv", index=False)
    importance_df.to_csv(HERE / "feature_importance.csv", index=False)
    coef_df.to_csv(HERE / "logreg_coefficients.csv", index=False)

    summary = {
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": int(x.shape[1]),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "metrics": metrics,
        "top_features_gb": importance_df.head(8).to_dict(orient="records"),
    }
    with open(HERE / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(metrics_df.to_string(index=False))
    print("\nTop feature importances (Gradient Boosting):")
    print(importance_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
