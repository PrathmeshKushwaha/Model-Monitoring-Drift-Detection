import csv
import pandas as pd
from pathlib import Path


COMMON_DELIMITERS = [",", ";", "\t", "|"]

EXPECTED_COLUMNS = {
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "duration", "campaign", "pdays", "previous",
    "poutcome", "y"
}

TARGET_COLUMN = "y"


NUMERICAL_FEATURES = [
    "age", "balance", "day", "duration",
    "campaign", "pdays", "previous"
]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]


def load_csv(path: str, sample_size: int = 5000) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Read sample for sniffing
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(sample_size)

    delimiter = None

    # 1. Try csv.Sniffer
    try:
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
    except csv.Error:
        pass

    # 2. Fallback to common delimiters
    if delimiter is None:
        for d in COMMON_DELIMITERS:
            try:
                df = pd.read_csv(path, sep=d)
                if df.shape[1] > 1:  # sanity check
                    delimiter = d
                    break
            except Exception:
                continue

    # 3. Hard fail if nothing works
    if delimiter is None:
        raise ValueError("Could not detect CSV delimiter")

    # 4. Load final dataframe
    df = pd.read_csv(path, sep=delimiter)

    print(f"[INFO] Loaded CSV using delimiter: '{delimiter}'")
    return df

def save_reference_data(df, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def validate_data(path: str) -> pd.DataFrame:
    df = load_csv(path)         

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    extra_cols = set(df.columns) - EXPECTED_COLUMNS

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if extra_cols:
        raise ValueError(f"Unexpected columns: {extra_cols}")

    if not set(df[TARGET_COLUMN].unique()).issubset({"yes", "no"}):
        raise ValueError("Invalid target values detected")

    if df.isnull().any().any():
        raise ValueError("Null values detected in dataset")

    return df


def split_features_target(df: pd.DataFrame):
    # Split dataframe into features and target, preserving feature types.
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"yes": 1, "no": 0})

    X_num = X[NUMERICAL_FEATURES]
    X_cat = X[CATEGORICAL_FEATURES]

    return X, y, X_num, X_cat