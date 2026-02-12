import pandas as pd
import numpy as np

INPUT_PATH = "data/production/batch_covariate.csv"
OUTPUT_PATH = "data/production/batch_feature_drift.csv"

df = pd.read_csv(INPUT_PATH)
drifted_df = df.copy()

# NUMERICAL DRIFT: shift distributions
num_cols = drifted_df.select_dtypes(include="number").columns

for col in num_cols:
    drifted_df[col] = drifted_df[col] * 1.3 + np.random.normal(0, 0.2, len(drifted_df))

# CATEGORICAL DRIFT: skew category frequencies
cat_cols = drifted_df.select_dtypes(exclude="number").columns

for col in cat_cols:
    if drifted_df[col].nunique() > 1:
        dominant = drifted_df[col].mode()[0]
        mask = np.random.rand(len(drifted_df)) < 0.4
        drifted_df.loc[mask, col] = dominant

drifted_df.to_csv(OUTPUT_PATH, index=False)

print("Feature-drifted production data saved to:", OUTPUT_PATH)
