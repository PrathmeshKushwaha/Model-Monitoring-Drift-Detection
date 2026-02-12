from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/production/pool.csv")
OUTPUT_PATH = Path("data/production/batch_covariate.csv")

TARGET_COLUMN = "y"


def simulate_covariate_drift(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.copy()

    # ---- Numerical drift ----
    # Age shift: older customers
    drifted["age"] = (drifted["age"] + 10).clip(18, 95)

    # Balance increase: wealthier customers
    drifted["balance"] = drifted["balance"] * 1.5

    # ---- Categorical drift ----
    # Increase proportion of 'retired'
    retired_mask = drifted["job"] == "retired"
    non_retired = drifted[~retired_mask]

    extra_retired = non_retired.sample(
        frac=0.15, random_state=42
    )
    drifted.loc[extra_retired.index, "job"] = "retired"

    # Reduce 'primary' education
    primary_mask = drifted["education"] == "primary"
    downgrade = drifted[primary_mask].sample(
        frac=0.3, random_state=42
    )
    drifted.loc[downgrade.index, "education"] = "secondary"

    return drifted


def main():
    df = pd.read_csv(INPUT_PATH)
    drifted_df = simulate_covariate_drift(df)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    drifted_df.to_csv(OUTPUT_PATH, index=False)

    print("[INFO] Covariate drift simulated")
    print(f"[INFO] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
