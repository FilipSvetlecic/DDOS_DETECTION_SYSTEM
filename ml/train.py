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
        n_estimators=100,    # 100 trees — good balance of accuracy vs speed on a laptop
        max_depth=20,        # limits tree size to keep memory usage manageable
        min_samples_leaf=5,  # avoids overfitting to tiny clusters of flows
        class_weight="balanced",  # compensates for the 60/40 DDoS/BENIGN imbalance
        n_jobs=-1,           # uses all CPU cores
        random_state=42,
    )
    clf.fit(X_train, y_train)

    return clf

def evaluate(clf, X_test, y_test):
    log.info("Evaluating on test set...")
 
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # probability of DDoS class
 
    report = classification_report(
        y_test, y_pred,
        target_names=["BENIGN (0)", "DDoS (1)"],
        digits=4,
    )
    cm      = confusion_matrix(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
 
    # Feature importances ranked highest to lowest
    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
 
    log.info(f"\n{report}")
    log.info(f"  ROC-AUC: {auc:.4f}")
    log.info("  Confusion matrix (rows=actual, cols=predicted):")
    log.info(f"              Pred BENIGN  Pred DDoS")
    log.info(f"  Act BENIGN  {cm[0][0]:>10,}  {cm[0][1]:>9,}")
    log.info(f"  Act DDoS    {cm[1][0]:>10,}  {cm[1][1]:>9,}")
    log.info("  Feature importances:")
    for name, score in importances:
        bar = "█" * int(score * 40)
        log.info(f"    {name:<22} {score:.4f}  {bar}")
 
    return report, cm, auc, importances
 

def save(clf):
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(clf, model_path)

def main():
    X_train, X_test, y_train, y_test = load_data()
    clf = train(X_train, y_train)
    evaluate(clf, X_test, y_test)
    save(clf)

if __name__ == "__main__":
    main()