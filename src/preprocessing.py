"""
preprocessing.py
-----------------
Data loading, cleaning, and feature engineering for the credit card
fraud dataset.

The dataset's V1-V28 columns are already PCA components (pre-scaled by
the original data providers), so they're left untouched. Only 'Time'
and 'Amount' are raw, so they get cleaned and engineered here.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

RAW_FEATURES = [f"V{i}" for i in range(1, 29)]
ENGINEERED_FEATURES = ["scaled_amount", "amount_log", "hour_of_day"]
MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def load_data(path):
    """Load the raw Kaggle creditcard.csv file."""
    return pd.read_csv(path)


def clean_data(df):
    """
    Drop exact duplicate rows.

    This dataset is known to contain ~1,081 duplicate transactions
    (likely re-logged entries). Leaving them in lets the same row end
    up in both the train and test split, which leaks information and
    inflates test-set performance, so they're removed before splitting.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"Removed {removed} duplicate rows ({before} -> {len(df)})")
    return df


def engineer_features(df):
    """
    Build model-ready features from the raw 'Time' and 'Amount' columns.

    - scaled_amount : Amount, robust-scaled (median/IQR) so large
      transaction outliers don't dominate distance-based parts of the
      pipeline (like the SMOTE neighbour search).
    - amount_log    : log1p(Amount), which compresses the long right
      tail of transaction amounts into a more model-friendly range.
    - hour_of_day    : 'Time' is seconds elapsed since the first
      transaction in the dataset (spans ~2 days). Converting it to an
      hour-of-day (0-23) captures cyclical spending patterns instead of
      a meaningless raw second-count.
    """
    df = df.copy()
    df["amount_log"] = np.log1p(df["Amount"])
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Stratified train/test split, preserving the fraud ratio in both sets."""
    X = df[RAW_FEATURES + ["Amount", "amount_log", "hour_of_day"]]
    y = df["Class"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def fit_scaler(X_train):
    """Fit a RobustScaler on the training 'Amount' column only (no leakage)."""
    scaler = RobustScaler()
    scaler.fit(X_train[["Amount"]])
    return scaler


def apply_scaler(X, scaler):
    """Apply a fitted scaler to produce the final 'scaled_amount' feature set."""
    X = X.copy()
    X["scaled_amount"] = scaler.transform(X[["Amount"]])
    return X[MODEL_FEATURES]
