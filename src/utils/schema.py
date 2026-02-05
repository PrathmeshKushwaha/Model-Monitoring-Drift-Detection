from typing import Dict, List

NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

TARGET_COLUMN = "y"

EXPECTED_COLUMNS = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
