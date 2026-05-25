import pandas as pd

X_train = pd.read_parquet("data/processed/X_train.parquet")
X_test  = pd.read_parquet("data/processed/X_test.parquet")
y_train = pd.read_parquet("data/processed/y_train.parquet")

print(X_train.shape)        # expect ~(90k+, 7)
print(X_train.isnull().sum())  # all zeros — no NaNs leaked through
print(X_train.describe())   # means near 0, stds near 1 — confirms scaling worked
print(y_train["label"].value_counts())  # check class balance