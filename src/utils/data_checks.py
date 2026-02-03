import pandas as pd


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


def load_and_validate_data(path: str) -> pd.DataFrame:
    # Load CSV and perform strict schema and target validation.
    df = pd.read_csv(path, sep=None, engine="python")
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