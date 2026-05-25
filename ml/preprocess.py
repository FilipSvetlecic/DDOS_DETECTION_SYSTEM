from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib




ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT/"data"/"raw"
PROCESSED = ROOT/"data"/"processed"
MODEL_DIR = ROOT/"ml"

RAW_CSV = RAW_DIR/"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

RAW_FEATURE_COLS = [
    " Flow Duration",
    " Total Fwd Packets",
    " Flow Bytes/s",
    " Flow Packets/s",
    " Average Packet Size",
    " SYN Flag Count",
    " ACK Flag Count"
]
LABEL_COL = " Label"

CLEAN_FEATURE_NAMES = [
    "flow_duration",
    "total_fwd_packets",
    "flow_bytes_s",
    "flow_pkts_s",
    "avg_packet_size",
    "syn_flag_count",
    "ack_flag_count"
]

def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    needed = [c.strip() for c in RAW_FEATURE_COLS] + ["Label"]
    df = df[needed].copy()
    df.columns = CLEAN_FEATURE_NAMES + ["label"]
    return df

# "BENIGN" - False - 0
# "DDoS" - True - 1
def clean_labels(df):
    df["label"] = df["label"].str.strip()
    df["label"] = (df["label"] != "BENIGN").astype(int)
    return df

def fix_inf_nan(df):
    features = CLEAN_FEATURE_NAMES

    inf_counts = np.isinf(df[features]).sum()
    if inf_counts.any():
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=features)
    return df

def remove_duplicates(df):
    df = df.drop_duplicates(subset=CLEAN_FEATURE_NAMES)
    return df

def split(df):
    test_size = 0.2
    random_state = 42
    X = df[CLEAN_FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=CLEAN_FEATURE_NAMES,
        index=X_train.index,
        )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=CLEAN_FEATURE_NAMES,
        index=X_test.index,
    )
    scaler_path = MODEL_DIR/"scaler.joblib"
    joblib.dump(scaler, scaler_path)
    return X_train_sc, X_test_sc, scaler

def save(X_train, X_test, y_train, y_test, scaler):
    X_train.to_parquet(PROCESSED / "X_train.parquet", index=False)
    X_test.to_parquet (PROCESSED / "X_test.parquet",  index=False)
    y_train.to_frame().to_parquet(PROCESSED / "y_train.parquet", index=False)
    y_test.to_frame() .to_parquet(PROCESSED / "y_test.parquet",  index=False)


def main():
    df = load_raw(RAW_CSV)
    df = clean_labels(df)
    df = fix_inf_nan(df)
    df = remove_duplicates(df)
    X_train, X_test, y_train, y_test = split(df)
    X_train_sc, X_test_sc, scaler = scale(X_train, X_test)
    save(X_train_sc, X_test_sc, y_train, y_test, scaler)

if __name__ == "__main__":
    main()
