import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.data_checks import validate_data

RAW_PATH = "data/raw/bank-full.csv"
DATA_DIR = Path("data")
PROD_DIR = DATA_DIR / "production"

DATA_DIR.mkdir(exist_ok=True)
PROD_DIR.mkdir(exist_ok=True)

# 1. Load + validate raw data
df = validate_data(RAW_PATH)

# 2. First split: train vs temp (60 / 40)
train_df, temp_df = train_test_split(
    df,
    test_size=0.4,
    random_state=42,
    stratify=df["y"]
)

# 3. Second split: test vs production pool (20 / 20)
test_df, prod_pool_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["y"]
)

# 4. Save datasets
train_df.to_csv(DATA_DIR / "train.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)
prod_pool_df.to_csv(PROD_DIR / "pool.csv", index=False)

# 5. Freeze reference (copy of train)
train_df.to_csv(DATA_DIR / "reference.csv", index=False)

print("Data split completed:")
print(f"Train: {len(train_df)}")
print(f"Test: {len(test_df)}")
print(f"Production pool: {len(prod_pool_df)}")
