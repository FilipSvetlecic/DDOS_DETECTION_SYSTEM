"""
app/predictor.py

Loads the scaler and model once at startup.
Exposes a single predict() function used by the middleware.
"""

import pandas as pd
from pathlib import Path
import joblib
import math

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "ml"

SCALER_PATH = MODEL_DIR / "scaler.joblib"
MODEL_PATH = MODEL_DIR / "model.joblib"

FEATURE_NAMES = [
    "flow_duration",
    "total_fwd_packets",
    "flow_bytes_s",
    "flow_pkts_s",
    "avg_packet_size",
    "syn_flag_count",
    "ack_flag_count",
]

# Loaded once when the module is imported — not on every request
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)


def predict(features: dict) -> dict:
    """
    Takes a dict of raw feature values, scales them, runs inference.
    Returns label (BENIGN / DDoS) and the model's confidence (0.0 – 1.0).
    """
    for k in FEATURE_NAMES:
        if not math.isfinite(features.get(k, 0.0)):
            features[k] = 0.0

    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=FEATURE_NAMES
    )
    label_int = model.predict(df_scaled)[0]
    confidence = model.predict_proba(df_scaled)[0][label_int]

    return {
        "label":      "DDoS" if label_int == 1 else "BENIGN",
        "confidence": round(float(confidence), 4),
    }