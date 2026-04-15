"""
Heart Disease Random Forest Model Trainer
Trains on the dataset columns: age, sex, cp, trestbps, chol, fbs,
restecg, thalach, exang, oldpeak, slope, ca, thal, spo2, temperature, target
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "spo2", "temperature"
]

def generate_synthetic_data(n=1200):
    """
    Generate synthetic training data that mirrors the CSV distribution
    when no real CSV is present. Replace with pd.read_csv() if you have the file.
    """
    np.random.seed(42)
    rows = []
    for _ in range(n):
        target = np.random.randint(0, 2)
        age        = np.random.randint(29, 77)
        sex        = np.random.randint(0, 2)
        cp         = np.random.randint(0, 4)
        trestbps   = np.random.randint(94, 200)
        chol       = np.random.randint(126, 564)
        fbs        = np.random.randint(0, 2)
        restecg    = np.random.randint(0, 3)
        thalach    = int(np.random.normal(150 - 8 * target, 22))
        exang      = int(np.random.random() < (0.55 if target else 0.14))
        oldpeak    = round(max(0, np.random.normal(1.5 * target + 0.3, 1.1)), 1)
        slope      = np.random.randint(0, 3)
        ca         = np.random.randint(0, 4)
        thal       = np.random.choice([0, 1, 2, 3])
        spo2       = round(np.random.normal(97.5 - 0.8 * target, 1.2), 1)
        temperature= round(np.random.normal(37.0 + 0.2 * target, 0.4), 1)
        rows.append([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                     exang, oldpeak, slope, ca, thal, spo2, temperature, target])
    return rows

def train():
    # ── Try to load real CSV first ──────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "heart_data.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        X = df[FEATURE_COLS].values
        y = df["target"].values
        print(f"[train] Loaded real CSV: {len(df)} rows")
    else:
        data = generate_synthetic_data(1200)
        X = np.array([r[:-1] for r in data])
        y = np.array([r[-1]  for r in data])
        print("[train] No CSV found — using synthetic data (place heart_data.csv in ml_service/ for real training)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    print(f"[train] Accuracy: {acc:.3f}")

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))

    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, "rf_model.pkl"),  "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, "scaler.pkl"),    "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "importances.pkl"), "wb") as f:
        pickle.dump(importances, f)

    print("[train] Saved rf_model.pkl, scaler.pkl, importances.pkl")
    return acc

if __name__ == "__main__":
    train()