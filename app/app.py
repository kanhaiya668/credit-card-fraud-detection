"""
app.py
------
Flask web application for the Credit Card Fraud Detection project.

Three tools in one app:
  1. Check a single transaction (with a "load a real sample" shortcut,
     since typing 28 anonymized PCA values by hand isn't realistic)
  2. Upload a CSV and bulk-check every row in it
  3. A dashboard summarizing the trained model's performance and
     patterns found in the training data

Run with:  python app/app.py   (from the project root)
Then open: http://127.0.0.1:5000
"""

import io
import json
import sys
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    session,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import predict_bulk, predict_single  # noqa: E402

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "app" / "static" / "data"
TMP_DIR = ROOT / "app" / "static" / "tmp"
TMP_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "fraud-detection-demo-secret-key"  # fine for a local demo app

# ---- Load model artifacts once at startup ---------------------------------
model = joblib.load(MODELS_DIR / "fraud_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")

with open(MODELS_DIR / "metrics.json") as f:
    METRICS = json.load(f)

with open(DATA_DIR / "sample_transactions.json") as f:
    SAMPLE_TRANSACTIONS = json.load(f)

with open(DATA_DIR / "dashboard_data.json") as f:
    DASHBOARD_DATA = json.load(f)

RAW_FEATURES = [f"V{i}" for i in range(1, 29)]


@app.context_processor
def inject_globals():
    """Make the active model name available in the sidebar on every page."""
    return {"active_model_name": METRICS["best_model"]}


# ---- Routes -----------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html", active_page="check")


@app.route("/api/sample")
def api_sample():
    """Return one random sample transaction for the 'load sample' button."""
    import random

    kind = request.args.get("type", "random")
    pool = SAMPLE_TRANSACTIONS
    if kind == "fraud":
        pool = [s for s in SAMPLE_TRANSACTIONS if s["true_label"] == "Fraud"]
    elif kind == "legit":
        pool = [s for s in SAMPLE_TRANSACTIONS if s["true_label"] == "Legit"]
    return jsonify(random.choice(pool))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Single-transaction prediction, called via fetch() from the check page."""
    payload = request.get_json()
    try:
        record = {f: float(payload[f]) for f in RAW_FEATURES}
        record["Amount"] = float(payload["Amount"])
        record["hour_of_day"] = float(payload.get("hour_of_day", 12))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid or missing field: {exc}"}), 400

    label, probability = predict_single(record, model, scaler)
    return jsonify({
        "prediction": label,
        "fraud_probability": round(probability, 4),
        "risk_level": risk_level(probability),
    })


def risk_level(probability):
    if probability >= 0.75:
        return "High"
    if probability >= 0.3:
        return "Medium"
    return "Low"


@app.route("/bulk", methods=["GET", "POST"])
def bulk():
    if request.method == "GET":
        return render_template("bulk.html", active_page="bulk", results=None)

    file = request.files.get("csv_file")
    if not file or file.filename == "":
        return render_template(
            "bulk.html", active_page="bulk", results=None,
            error="Please choose a CSV file first."
        )

    try:
        df = pd.read_csv(file)
        result = predict_bulk(df, model, scaler)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
        return render_template(
            "bulk.html", active_page="bulk", results=None,
            error=f"Couldn't process that file: {exc}"
        )

    token = uuid.uuid4().hex[:10]
    result_path = TMP_DIR / f"results_{token}.csv"
    result.to_csv(result_path, index=False)
    session["last_result_file"] = result_path.name

    summary = {
        "total_rows": len(result),
        "fraud_flagged": int((result["prediction"] == "Fraud").sum()),
        "legit_flagged": int((result["prediction"] == "Legit").sum()),
    }
    summary["fraud_pct"] = round(summary["fraud_flagged"] / summary["total_rows"] * 100, 2)

    preview_cols = ["Amount", "hour_of_day", "fraud_probability", "prediction"]
    preview_cols = [c for c in preview_cols if c in result.columns]
    preview_rows = result[preview_cols].head(25).to_dict(orient="records")

    return render_template(
        "bulk.html",
        active_page="bulk",
        results={"summary": summary, "rows": preview_rows, "showing": len(preview_rows)},
        error=None,
    )


@app.route("/bulk/download")
def bulk_download():
    filename = session.get("last_result_file")
    if not filename:
        return "No results to download yet -- run a bulk check first.", 404
    path = TMP_DIR / filename
    if not path.exists():
        return "That results file has expired -- please run the check again.", 404
    return send_file(path, as_attachment=True, download_name="fraud_check_results.csv")


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        data=DASHBOARD_DATA,
        metrics=METRICS,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
