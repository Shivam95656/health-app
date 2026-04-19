"""
ecg_predict.py
==============
ECG-ONLY heart disease risk predictor.
Called by ecg.routes.js via child_process — reads JSON from stdin, writes JSON to stdout.

Usage (from Node):
  python ecg_predict.py '{"heart_rate": 72, "label": "Normal", "sample_rate": 360, "duration_sec": 10, "samples": [...]}'

Input  (JSON):
  heart_rate   – bpm from ECG record
  label        – CNN arrhythmia label ("Normal", "AFIB", "ST-elevation", etc.)
  sample_rate  – Hz
  duration_sec – seconds
  samples      – raw float array (used for signal feature extraction)

Output (JSON):
  risk_percentage, risk_level, no_disease_pct,
  ecg_label, ecg_features, disease_breakdown,
  flagged_features, top_features
"""

import sys, json, os, pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load ECG-aware model ──────────────────────────────────────────────────────
def load_models():
    ecg_model_path = os.path.join(BASE, "ecg_rf_model.pkl")
    # Auto-train if missing
    if not os.path.exists(ecg_model_path):
        sys.stderr.write("[ecg_predict] Model missing — training now...\n")
        from train_ecg_model import train
        train()

    with open(os.path.join(BASE, "ecg_rf_model.pkl"),    "rb") as f: model       = pickle.load(f)
    with open(os.path.join(BASE, "ecg_scaler.pkl"),      "rb") as f: scaler      = pickle.load(f)
    with open(os.path.join(BASE, "ecg_importances.pkl"), "rb") as f: importances = pickle.load(f)
    with open(os.path.join(BASE, "ecg_label_map.pkl"),   "rb") as f: label_map   = pickle.load(f)
    return model, scaler, importances, label_map

# ── ECG label encoding ────────────────────────────────────────────────────────
ECG_RISK_WEIGHT = {
    0: 0.0,   # Normal
    1: 0.7,   # AFIB
    2: 1.0,   # ST-elevation
    3: 0.8,   # ST-depression
    4: 0.6,   # T-wave inversion
    5: 0.5,   # PVC
    6: 0.65,  # SVT
    7: 0.55,  # LVH
    8: 0.75,  # LBBB
}

LABEL_TO_CP      = {"st-elevation": 3, "st-depression": 2, "t-wave inversion": 1}
LABEL_TO_EXANG   = {"st-elevation": 1, "st-depression": 1}
LABEL_TO_OLDPEAK = {"st-elevation": 2.5, "st-depression": 1.8, "t-wave inversion": 1.0}
LABEL_TO_RESTECG = {"normal": 0, "lbbb": 2, "lvh": 1, "st-elevation": 2, "st-depression": 1}

FEATURE_LABELS = {
    "age":            "Age",
    "sex":            "Sex",
    "cp":             "Chest Pain Type",
    "trestbps":       "Resting Blood Pressure",
    "chol":           "Cholesterol",
    "fbs":            "Fasting Blood Sugar",
    "restecg":        "Resting ECG",
    "thalach":        "Heart Rate (ECG)",
    "exang":          "Exercise Induced Angina",
    "oldpeak":        "ST Depression",
    "slope":          "ST Slope",
    "ca":             "Major Vessels",
    "thal":           "Thalassemia",
    "spo2":           "Blood Oxygen (SpO2)",
    "temperature":    "Body Temperature",
    "ecg_label_code": "ECG Arrhythmia Class",
    "hr_deviation":   "Heart Rate Deviation",
}

RISK_FLAGS = {
    "thalach":        lambda v: v < 60 or v > 150,
    "oldpeak":        lambda v: v > 2.0,
    "ecg_label_code": lambda v: v > 0,
    "hr_deviation":   lambda v: abs(v) > 0.20,
    "restecg":        lambda v: v > 0,
    "exang":          lambda v: v == 1,
}

# ── Signal feature extraction ─────────────────────────────────────────────────
def extract_signal_features(samples, sample_rate):
    """Extract meaningful features from raw ECG sample array."""
    if not samples or len(samples) < 10:
        return {}

    arr = np.array(samples, dtype=float)
    sr  = max(sample_rate or 360, 1)

    features = {
        "signal_mean":    round(float(np.mean(arr)), 4),
        "signal_std":     round(float(np.std(arr)), 4),
        "signal_min":     round(float(np.min(arr)), 4),
        "signal_max":     round(float(np.max(arr)), 4),
        "signal_range":   round(float(np.max(arr) - np.min(arr)), 4),
        "signal_rms":     round(float(np.sqrt(np.mean(arr**2))), 4),
        "total_samples":  len(arr),
        "duration_sec":   round(len(arr) / sr, 2),
    }

    # Zero-crossing rate (arrhythmia indicator)
    zero_crossings = np.sum(np.diff(np.sign(arr - np.mean(arr))) != 0)
    features["zero_crossing_rate"] = round(float(zero_crossings) / len(arr), 4)

    # Peak detection (simple threshold)
    threshold = np.mean(arr) + 0.5 * np.std(arr)
    peaks = np.where((arr[1:-1] > threshold) &
                     (arr[1:-1] > arr[:-2]) &
                     (arr[1:-1] > arr[2:]))[0]
    features["peak_count"] = int(len(peaks))

    # Estimated HR from peaks if heart_rate not provided
    if len(peaks) >= 2:
        avg_rr_samples = np.mean(np.diff(peaks))
        features["estimated_hr_from_peaks"] = round(float(sr * 60 / avg_rr_samples), 1)

    return features


# ── Core prediction ───────────────────────────────────────────────────────────
def predict_from_ecg(ecg_input: dict) -> dict:
    model, scaler, importances, label_map = load_models()

    label      = (ecg_input.get("label") or "Normal").strip()
    label_key  = label.lower()
    heart_rate = float(ecg_input.get("heart_rate") or 75)
    sample_rate= int(ecg_input.get("sample_rate") or 360)
    duration   = float(ecg_input.get("duration_sec") or 10)
    samples    = ecg_input.get("samples") or []

    # Encode label
    ecg_label_code = label_map.get(label_key, 0)
    ecg_risk_w     = ECG_RISK_WEIGHT.get(ecg_label_code, 0.0)

    # HR deviation from age-predicted max (use population avg age 50)
    age = 50
    predicted_max_hr = 220 - age
    hr_deviation = round((heart_rate - predicted_max_hr) / predicted_max_hr, 4)

    # Derive clinical proxies purely from ECG
    restecg  = LABEL_TO_RESTECG.get(label_key, 0)
    cp       = LABEL_TO_CP.get(label_key, 0)
    exang    = LABEL_TO_EXANG.get(label_key, 0)
    oldpeak  = LABEL_TO_OLDPEAK.get(label_key, 0.0)

    # Feature vector — 17 features matching ecg_rf_model
    feature_vector = [
        age,            # age (population avg)
        1,              # sex (population avg)
        cp,             # derived from ECG label
        120,            # trestbps (population avg)
        200,            # chol (population avg)
        0,              # fbs (population avg)
        restecg,        # derived from ECG label
        heart_rate,     # thalach — from ECG
        exang,          # derived from ECG label
        oldpeak,        # derived from ECG label
        1,              # slope (normal default)
        0,              # ca (normal default)
        2,              # thal (normal default)
        97.5,           # spo2 (normal default)
        37.0,           # temperature (normal default)
        ecg_label_code, # ECG arrhythmia code — KEY ECG FEATURE
        hr_deviation,   # HR deviation — KEY ECG FEATURE
    ]

    FEATURE_COLS = [
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal","spo2","temperature",
        "ecg_label_code","hr_deviation",
    ]

    x  = np.array([feature_vector])
    xs = scaler.transform(x)

    proba       = model.predict_proba(xs)[0]
    risk_pct    = round(float(proba[1]) * 100, 1)
    no_risk_pct = round(float(proba[0]) * 100, 1)

    vals = dict(zip(FEATURE_COLS, feature_vector))

    # ── Disease breakdown ─────────────────────────────────────────────────
    def sigmoid(z): return float(1 / (1 + np.exp(-z)))

    disease_breakdown = [
        {"name": "Coronary Artery Disease",
         "pct": round(sigmoid(vals["ca"]*0.6 + cp*0.4 + oldpeak*0.4 + ecg_risk_w*0.6 - 1.2)*100, 1)},
        {"name": "Arrhythmia",
         "pct": round(sigmoid(restecg*0.9 + ecg_label_code*0.35 + ecg_risk_w*0.9 - 0.5)*100, 1)},
        {"name": "Heart Failure",
         "pct": round(sigmoid(oldpeak*0.5 + (100 - heart_rate)*0.015 + exang*0.7 + ecg_risk_w*0.4 - 0.8)*100, 1)},
        {"name": "Valvular Heart Disease",
         "pct": round(sigmoid(vals["thal"]*0.5 + vals["ca"]*0.4 + ecg_risk_w*0.3 - 1.0)*100, 1)},
        {"name": "Hypertensive Heart Disease",
         "pct": round(sigmoid((vals["trestbps"]-120)*0.04 + ecg_risk_w*0.2 - 0.5)*100, 1)},
    ]

    # ── Risk flags ────────────────────────────────────────────────────────
    flagged = []
    for feat, check in RISK_FLAGS.items():
        val = vals.get(feat)
        if val is not None and check(val):
            flagged.append({
                "feature":    feat,
                "label":      FEATURE_LABELS.get(feat, feat),
                "value":      val,
                "importance": round(importances.get(feat, 0) * 100, 1),
            })
    flagged.sort(key=lambda x: -x["importance"])

    # ── Top features ──────────────────────────────────────────────────────
    top_features = sorted(
        [{"feature": k, "label": FEATURE_LABELS.get(k, k),
          "importance": round(v * 100, 1)} for k, v in importances.items()],
        key=lambda x: -x["importance"]
    )[:6]

    risk_level = (
        "Low"      if risk_pct < 30 else
        "Moderate" if risk_pct < 60 else
        "High"     if risk_pct < 80 else
        "Critical"
    )

    # ── Signal features from raw samples ─────────────────────────────────
    signal_features = extract_signal_features(samples, sample_rate)

    return {
        "risk_percentage":   risk_pct,
        "no_disease_pct":    no_risk_pct,
        "risk_level":        risk_level,
        "ecg_label":         label,
        "ecg_label_code":    ecg_label_code,
        "hr_deviation":      hr_deviation,
        "ecg_features":      signal_features,
        "disease_breakdown": disease_breakdown,
        "flagged_features":  flagged,
        "top_features":      top_features,
        "input_values":      vals,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        # Read JSON from first CLI arg or stdin
        if len(sys.argv) > 1:
            ecg_input = json.loads(sys.argv[1])
        else:
            ecg_input = json.loads(sys.stdin.read())

        result = predict_from_ecg(ecg_input)
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
