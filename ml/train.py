import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT/"data"/"processed"
MODEL_DIR = ROOT/"ml"
 
FEATURE_NAMES = [
    "flow_duration",
    "total_fwd_packets",
    "flow_bytes_s",
    "flow_pkts_s",
    "avg_packet_size",
    "syn_flag_count",
    "ack_flag_count",
]

def load_data():
    X_train = pd.read_parquet(PROCESSED / "X_train.parquet")
    X_test  = pd.read_parquet(PROCESSED / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(PROCESSED / "y_test.parquet").squeeze()
 
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,           
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf

def save(clf):
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(clf, model_path)

def main():
    X_train, X_test, y_train, y_test = load_data()
    clf = train(X_train, y_train)
    save(clf)

if __name__ == "__main__":
    main()