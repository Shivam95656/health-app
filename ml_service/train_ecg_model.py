"""
train_ecg_model.py
==================
Retrains the heart-disease Random Forest using the real heart_data.csv
PLUS synthetic ECG signal features (heart_rate / thalach, ecg_label_code).

New feature set (17 features):
  age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
  slope, ca, thal, spo2, temperature,
  ecg_label_code,   ← CNN arrhythmia class encoded as int
  hr_deviation      ← how far heart_rate deviates from age-predicted max

Saves:
  ecg_rf_model.pkl   – RandomForest classifier
  ecg_scaler.pkl     – StandardScaler
  ecg_importances.pkl– feature importance dict
  ecg_label_map.pkl  – label → int mapping
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(os.path.abspath(__file__))

# ── ECG label encoding ────────────────────────────────────────────────────────
ECG_LABEL_MAP = {
    "normal":           0,
    "afib":             1,
    "st-elevation":     2,
    "st-depression":    3,
    "t-wave inversion": 4,
    "pvc":              5,
    "svt":              6,
    "lvh":              7,
    "lbbb":             8,
}

# Risk weight per ECG label (higher = more cardiac risk signal)
ECG_RISK_WEIGHT = {
    0: 0.0,   # Normal
    1: 0.7,   # AFIB
    2: 1.0,   # ST-elevation (most dangerous)
    3: 0.8,   # ST-depression
    4: 0.6,   # T-wave inversion
    5: 0.5,   # PVC
    6: 0.65,  # SVT
    7: 0.55,  # LVH
    8: 0.75,  # LBBB
}

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "spo2", "temperature",
    "ecg_label_code",
    "hr_deviation",
]

FEATURE_LABELS = {
    "age":            "Age",
    "sex":            "Sex",
    "cp":             "Chest Pain Type",
    "trestbps":       "Resting Blood Pressure",
    "chol":           "Cholesterol",
    "fbs":            "Fasting Blood Sugar > 120",
    "restecg":        "Resting ECG",
    "thalach":        "Max Heart Rate (ECG)",
    "exang":          "Exercise Induced Angina",
    "oldpeak":        "ST Depression (Oldpeak)",
    "slope":          "Slope of ST Segment",
    "ca":             "Major Vessels (CA)",
    "thal":           "Thalassemia",
    "spo2":           "Blood Oxygen (SpO₂)",
    "temperature":    "Body Temperature",
    "ecg_label_code": "ECG Arrhythmia Class",
    "hr_deviation":   "Heart Rate Deviation",
}


def encode_ecg_label(label: str) -> int:
    """Convert CNN label string → int code."""
    return ECG_LABEL_MAP.get((label or "normal").lower().strip(), 0)


def hr_deviation(heart_rate: float, age: float) -> float:
    """
    How far the measured HR deviates from the age-predicted maximum HR.
    Positive = above predicted max (tachycardia signal).
    Negative = below (bradycardia signal).
    """
    predicted_max = 220 - age
    return round((heart_rate - predicted_max) / max(predicted_max, 1), 4)


def augment_with_ecg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic ECG columns to the base CSV during training.
    In real inference these come from the actual ECG record.
    During training we synthesise them so the model learns the feature space.
    """
    np.random.seed(42)
    n = len(df)
    target = df["target"].values

    # Simulate ECG label: diseased patients get higher-risk labels more often
    label_codes = []
    for t in target:
        if t == 1:  # disease
            label_codes.append(np.random.choice(
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                p=[0.15, 0.12, 0.15, 0.15, 0.12, 0.08, 0.10, 0.08, 0.05]
            ))
        else:       # no disease
            label_codes.append(np.random.choice(
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                p=[0.65, 0.06, 0.04, 0.05, 0.06, 0.04, 0.04, 0.03, 0.03]
            ))

    df = df.copy()
    df["ecg_label_code"] = label_codes
    df["hr_deviation"]   = df.apply(
        lambda r: hr_deviation(r["thalach"], r["age"]), axis=1
    )
    return df


def train():
    csv_path = os.path.join(BASE, "heart_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"heart_data.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[train_ecg] Loaded {len(df)} rows from heart_data.csv")

    # Add ECG features
    df = augment_with_ecg_features(df)

    X = df[FEATURE_COLS].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Random Forest (primary model) ─────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    acc   = accuracy_score(y_test, rf.predict(X_test_s))
    proba = rf.predict_proba(X_test_s)[:, 1]
    auc   = roc_auc_score(y_test, proba)

    print(f"[train_ecg] Accuracy : {acc:.3f}")
    print(f"[train_ecg] ROC-AUC  : {auc:.3f}")
    print(classification_report(y_test, rf.predict(X_test_s),
                                 target_names=["No Disease", "Disease"]))

    # 5-fold CV
    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=5, scoring="roc_auc")
    print(f"[train_ecg] 5-fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    importances = dict(zip(FEATURE_COLS, rf.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])
    print("\n[train_ecg] Top feature importances:")
    for feat, imp in top[:8]:
        print(f"  {FEATURE_LABELS[feat]:<35} {imp:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────
    with open(os.path.join(BASE, "ecg_rf_model.pkl"),    "wb") as f: pickle.dump(rf,           f)
    with open(os.path.join(BASE, "ecg_scaler.pkl"),      "wb") as f: pickle.dump(scaler,       f)
    with open(os.path.join(BASE, "ecg_importances.pkl"), "wb") as f: pickle.dump(importances,  f)
    with open(os.path.join(BASE, "ecg_label_map.pkl"),   "wb") as f: pickle.dump(ECG_LABEL_MAP,f)

    print("\n[train_ecg] ✓ Saved ecg_rf_model.pkl, ecg_scaler.pkl, ecg_importances.pkl, ecg_label_map.pkl")
    return {"accuracy": acc, "auc": auc}


if __name__ == "__main__":
    train()
