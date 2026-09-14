"""
train_pipeline.py
------------------
End-to-end training pipeline for the Credit Card Fraud Detection project.

Steps:
    1. Load raw data
    2. Clean (drop duplicates)
    3. Engineer features (scaled amount, log amount, hour of day)
    4. Stratified train/test split
    5. Apply custom SMOTE to the TRAINING set only (never touch test data,
       or the evaluation numbers become meaningless)
    6. Train and compare 3 candidate models
    7. Pick the best model by PR-AUC (average precision) -- for a dataset
       that's 99.83% legitimate transactions, plain accuracy or even
       ROC-AUC can look great while still missing most fraud, so PR-AUC
       (which focuses on the positive/minority class) is the right metric
       to select on here.
    8. Save the best model, the fitted scaler, and a metrics report

Run with:  python src/train_pipeline.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from custom_smote import CustomSMOTE
from preprocessing import (
    apply_scaler,
    clean_data,
    engineer_features,
    fit_scaler,
    load_data,
    split_data,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42


def build_candidate_models():
    """
    Three deliberately different model families, so the comparison is
    meaningful rather than three flavors of the same algorithm:

    - Logistic Regression : simple, fast, fully interpretable baseline.
    - Random Forest        : bagged trees, robust to noisy/irrelevant
      features, handles non-linear boundaries well.
    - HistGradientBoosting : scikit-learn's native histogram-based
      gradient boosting (the same family of algorithm as XGBoost/
      LightGBM). Usually the strongest performer on tabular fraud data,
      included here as the "production-grade" candidate.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=8,
            learning_rate=0.08,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    """Score a fitted model on the untouched test set and return a metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {metrics['roc_auc']}  |  PR-AUC: {metrics['pr_auc']}")

    return metrics


def run_pipeline():
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Fitting scaler on training data...")
    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_test = apply_scaler(X_test, scaler)

    fraud_before = int(y_train.sum())
    print(f"\nTraining set before SMOTE: {len(y_train)} rows, {fraud_before} fraud "
          f"({fraud_before / len(y_train):.3%})")

    print("Applying custom SMOTE to training data only...")
    smote = CustomSMOTE(sampling_strategy=0.15, k_neighbors=5, random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train.values, y_train.values)

    fraud_after = int(y_train_res.sum())
    print(f"Training set after SMOTE:  {len(y_train_res)} rows, {fraud_after} fraud "
          f"({fraud_after / len(y_train_res):.3%})")

    print("\nTraining candidate models...")
    models = build_candidate_models()
    all_metrics = []
    fitted_models = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train_res, y_train_res)
        elapsed = time.time() - start
        print(f"\n{name} trained in {elapsed:.1f}s")

        metrics = evaluate_model(name, model, X_test.values, y_test.values)
        metrics["train_seconds"] = round(elapsed, 1)
        all_metrics.append(metrics)
        fitted_models[name] = model

    best = max(all_metrics, key=lambda m: m["pr_auc"])
    best_name = best["model"]
    best_model = fitted_models[best_name]

    print(f"\n{'=' * 50}")
    print(f"Best model: {best_name}  (PR-AUC = {best['pr_auc']})")
    print(f"{'=' * 50}")

    joblib.dump(best_model, MODELS_DIR / "fraud_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    report = {
        "best_model": best_name,
        "feature_order": list(X_train.columns),
        "dataset_rows_after_cleaning": len(df),
        "fraud_rate_full_dataset": round(df["Class"].mean(), 6),
        "train_rows_after_smote": len(y_train_res),
        "test_rows": len(y_test),
        "model_comparison": all_metrics,
        "total_pipeline_seconds": round(time.time() - t0, 1),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved model -> {MODELS_DIR / 'fraud_model.pkl'}")
    print(f"Saved scaler -> {MODELS_DIR / 'scaler.pkl'}")
    print(f"Saved metrics -> {MODELS_DIR / 'metrics.json'}")
    print(f"Total pipeline time: {report['total_pipeline_seconds']}s")

    return report


if __name__ == "__main__":
    run_pipeline()
