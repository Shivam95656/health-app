"""
Heart Disease Prediction Microservice  –  runs on port 5001
Endpoints:
  POST /predict   { age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal, spo2, temperature }
  GET  /health
"""

import os, pickle, json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Auto-train if model not present ──────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(BASE, "rf_model.pkl")):
    print("[service] Model not found – training now …")
    from train_model import train
    train()

with open(os.path.join(BASE, "rf_model.pkl"),    "rb") as f: model       = pickle.load(f)
with open(os.path.join(BASE, "scaler.pkl"),       "rb") as f: scaler      = pickle.load(f)
with open(os.path.join(BASE, "importances.pkl"),  "rb") as f: importances = pickle.load(f)

FEATURE_COLS = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","spo2","temperature"
]

FEATURE_LABELS = {
    "age":       "Age",
    "sex":       "Sex",
    "cp":        "Chest Pain Type",
    "trestbps":  "Resting Blood Pressure",
    "chol":      "Cholesterol",
    "fbs":       "Fasting Blood Sugar > 120",
    "restecg":   "Resting ECG",
    "thalach":   "Max Heart Rate",
    "exang":     "Exercise Induced Angina",
    "oldpeak":   "ST Depression (Oldpeak)",
    "slope":     "Slope of ST Segment",
    "ca":        "Major Vessels (CA)",
    "thal":      "Thalassemia",
    "spo2":      "Blood Oxygen (SpO₂)",
    "temperature":"Body Temperature"
}

# ── Reference ranges for risk flagging ───────────────────────────────────
RISK_FLAGS = {
    "trestbps":  lambda v: v > 140,
    "chol":      lambda v: v > 240,
    "fbs":       lambda v: v == 1,
    "thalach":   lambda v: v < 100,
    "exang":     lambda v: v == 1,
    "oldpeak":   lambda v: v > 2.0,
    "ca":        lambda v: v >= 2,
    "spo2":      lambda v: v < 95,
    "temperature": lambda v: v > 37.5,
}

app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # ── Build feature vector ──────────────────────────────────────────────
    try:
        x = np.array([[float(data[col]) for col in FEATURE_COLS]])
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400

    xs = scaler.transform(x)

    # ── RF probability ────────────────────────────────────────────────────
    proba       = model.predict_proba(xs)[0]          # [P(no disease), P(disease)]
    risk_pct    = round(float(proba[1]) * 100, 1)
    no_risk_pct = round(float(proba[0]) * 100, 1)

    # ── Per-condition disease-type risk breakdown ─────────────────────────
    # Based on medical literature mapping RF features → heart disease subtypes
    vals = {col: float(data[col]) for col in FEATURE_COLS}

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def cad_score():
        z = (vals["age"] - 55) * 0.04 + vals["chol"] * 0.005 + \
            vals["ca"] * 0.6 + vals["cp"] * 0.3 + \
            vals["trestbps"] * 0.01 + vals["oldpeak"] * 0.35
        return round(sigmoid(z - 1.2) * 100, 1)

    def arrhythmia_score():
        z = vals["restecg"] * 0.9 + (1 - vals["exang"]) * 0.3 + \
            (vals["thalach"] - 140) * (-0.02) + vals["fbs"] * 0.4
        return round(sigmoid(z - 0.5) * 100, 1)

    def heart_failure_score():
        z = vals["oldpeak"] * 0.5 + (100 - vals["thalach"]) * 0.025 + \
            vals["exang"] * 0.7 + (vals["age"] - 60) * 0.03 + \
            vals["slope"] * 0.2
        return round(sigmoid(z - 0.8) * 100, 1)

    def valvular_score():
        z = vals["thal"] * 0.55 + vals["ca"] * 0.4 + \
            (vals["age"] - 50) * 0.035 + vals["cp"] * 0.2
        return round(sigmoid(z - 1.0) * 100, 1)

    def hypertensive_score():
        z = (vals["trestbps"] - 120) * 0.04 + \
            (vals["chol"] - 200) * 0.003 + \
            vals["fbs"] * 0.5 + (vals["age"] - 50) * 0.025
        return round(sigmoid(z - 0.5) * 100, 1)

    disease_breakdown = [
        {"name": "Coronary Artery Disease",     "pct": cad_score()},
        {"name": "Arrhythmia",                  "pct": arrhythmia_score()},
        {"name": "Heart Failure",               "pct": heart_failure_score()},
        {"name": "Valvular Heart Disease",       "pct": valvular_score()},
        {"name": "Hypertensive Heart Disease",  "pct": hypertensive_score()},
    ]

    # ── Risk-flagged features ─────────────────────────────────────────────
    flagged = []
    for feat, check in RISK_FLAGS.items():
        if check(vals[feat]):
            flagged.append({
                "feature": feat,
                "label":   FEATURE_LABELS[feat],
                "value":   vals[feat],
                "importance": round(importances.get(feat, 0) * 100, 1)
            })
    flagged.sort(key=lambda x: -x["importance"])

    # ── Top feature importances ───────────────────────────────────────────
    top_features = sorted(
        [{"feature": k, "label": FEATURE_LABELS[k],
          "importance": round(v * 100, 1)} for k, v in importances.items()],
        key=lambda x: -x["importance"]
    )[:6]

    risk_level = (
        "Low"      if risk_pct < 30  else
        "Moderate" if risk_pct < 60  else
        "High"     if risk_pct < 80  else
        "Critical"
    )

    return jsonify({
        "risk_percentage":    risk_pct,
        "no_disease_pct":     no_risk_pct,
        "risk_level":         risk_level,
        "disease_breakdown":  disease_breakdown,
        "flagged_features":   flagged,
        "top_features":       top_features,
        "input_values":       vals
    })


if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", 5001))
    print(f"[service] Starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)