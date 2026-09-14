"""
predict.py
----------
Shared prediction helper. Loads the trained model + scaler once and
exposes simple functions the Flask app calls for both the single-
transaction form and the bulk CSV upload.
"""

import numpy as np
import pandas as pd

RAW_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_ORDER = RAW_FEATURES + ["scaled_amount", "amount_log", "hour_of_day"]


def build_feature_row(record, scaler):
    """
    Turn a single dict of {V1..V28, Amount, hour_of_day} into the exact
    feature vector the model was trained on (same order, same derived
    features as training).
    """
    amount = float(record["Amount"])
    row = {f: float(record[f]) for f in RAW_FEATURES}
    amount_df = pd.DataFrame({"Amount": [amount]})
    row["scaled_amount"] = float(scaler.transform(amount_df)[0][0])
    row["amount_log"] = float(np.log1p(amount))
    row["hour_of_day"] = float(record.get("hour_of_day", 12))
    return pd.DataFrame([row])[FEATURE_ORDER]


def predict_single(record, model, scaler):
    """Return (label:str, fraud_probability:float) for one transaction dict."""
    X = build_feature_row(record, scaler)
    proba = float(model.predict_proba(X.values)[0][1])
    label = "Fraud" if proba >= 0.5 else "Legit"
    return label, proba


def predict_bulk(df, model, scaler):
    """
    Run predictions for every row of an uploaded CSV.

    Expects columns V1..V28 and Amount. 'hour_of_day' is optional --
    if the CSV doesn't include it (e.g. it's just raw Kaggle-format
    data with 'Time' instead), it's derived from 'Time' or defaulted
    to noon.
    """
    df = df.copy()

    missing = [c for c in RAW_FEATURES + ["Amount"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if "hour_of_day" not in df.columns:
        if "Time" in df.columns:
            df["hour_of_day"] = (df["Time"] // 3600) % 24
        else:
            df["hour_of_day"] = 12

    df["scaled_amount"] = scaler.transform(df[["Amount"]])
    df["amount_log"] = np.log1p(df["Amount"])

    X = df[FEATURE_ORDER]
    probabilities = model.predict_proba(X.values)[:, 1]
    predictions = np.where(probabilities >= 0.5, "Fraud", "Legit")

    result = df.copy()
    result["fraud_probability"] = np.round(probabilities, 4)
    result["prediction"] = predictions
    return result
