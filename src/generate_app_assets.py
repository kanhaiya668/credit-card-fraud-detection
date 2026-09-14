"""
generate_app_assets.py
-----------------------
Builds two small JSON files the Flask app uses so it's actually usable
without forcing someone to hand-type 30 feature values:

  1. sample_transactions.json  - a handful of real transactions (mix of
     fraud + legit) pulled from the held-out test split, so the "Try a
     sample transaction" button on the single-check page can instantly
     fill the form with real data.
  2. dashboard_data.json       - pre-computed aggregate stats about the
     full dataset (fraud rate, amount patterns, hourly pattern) so the
     dashboard has real charts to show instead of empty placeholders.

Run with: python src/generate_app_assets.py  (after train_pipeline.py)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import clean_data, engineer_features, load_data, split_data

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
APP_DATA_DIR = ROOT / "app" / "static" / "data"
RAW_FEATURES = [f"V{i}" for i in range(1, 29)]


def build_sample_transactions(X_test, y_test, n_per_class=6):
    """Grab a few real fraud + legit rows from the test set for demo use."""
    samples = []
    for label in [0, 1]:
        idx = y_test[y_test == label].index
        chosen = np.random.RandomState(7).choice(idx, size=n_per_class, replace=False)
        for i in chosen:
            row = X_test.loc[i]
            record = {f: float(row[f]) for f in RAW_FEATURES}
            record["Amount"] = float(row["Amount"])
            record["hour_of_day"] = int(row["hour_of_day"])
            record["true_label"] = "Fraud" if label == 1 else "Legit"
            samples.append(record)
    np.random.RandomState(7).shuffle(samples)
    return samples


def build_dashboard_data(df, metrics_path):
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    hourly = (
        df.groupby("hour_of_day")["Class"]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "fraud_count"})
    )
    hourly_pattern = [
        {
            "hour": int(h),
            "total": int(row["total"]),
            "fraud_count": int(row["fraud_count"]),
            "fraud_rate": round(row["fraud_count"] / row["total"], 6),
        }
        for h, row in hourly.iterrows()
    ]

    amount_bins = [0, 10, 50, 100, 500, 1000, np.inf]
    bin_labels = ["0-10", "10-50", "50-100", "100-500", "500-1000", "1000+"]
    df["_amount_bucket"] = pd.cut(df["Amount"], bins=amount_bins, labels=bin_labels)
    amount_distribution = (
        df.groupby("_amount_bucket", observed=True)["Class"]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "fraud_count"})
    )
    amount_distribution = [
        {"bucket": b, "total": int(row["total"]), "fraud_count": int(row["fraud_count"])}
        for b, row in amount_distribution.iterrows()
    ]

    with open(metrics_path) as f:
        model_metrics = json.load(f)

    dashboard = {
        "total_transactions": int(len(df)),
        "fraud_count": int(len(fraud)),
        "legit_count": int(len(legit)),
        "fraud_rate_pct": round(len(fraud) / len(df) * 100, 4),
        "avg_fraud_amount": round(float(fraud["Amount"].mean()), 2),
        "avg_legit_amount": round(float(legit["Amount"].mean()), 2),
        "median_fraud_amount": round(float(fraud["Amount"].median()), 2),
        "median_legit_amount": round(float(legit["Amount"].median()), 2),
        "max_fraud_amount": round(float(fraud["Amount"].max()), 2),
        "hourly_pattern": hourly_pattern,
        "amount_distribution": amount_distribution,
        "model_metrics": model_metrics,
    }
    return dashboard


def main():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading + preparing data...")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("Building sample transactions for the demo form...")
    samples = build_sample_transactions(X_test, y_test)
    with open(APP_DATA_DIR / "sample_transactions.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} sample transactions.")

    print("Building dashboard aggregate data...")
    dashboard = build_dashboard_data(df, ROOT / "models" / "metrics.json")
    with open(APP_DATA_DIR / "dashboard_data.json", "w") as f:
        json.dump(dashboard, f, indent=2)
    print("Saved dashboard_data.json")


if __name__ == "__main__":
    main()
