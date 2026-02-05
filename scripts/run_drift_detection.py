from pathlib import Path
from src.monitoring.drift_detection import detect_feature_drift

REFERENCE_PATH = Path("data/reference.csv")
PRODUCTION_PATH = Path("data/production/pool.csv")
TARGET_COLUMN = "y"

if __name__ == "__main__":
    report = detect_feature_drift(
        reference_path=REFERENCE_PATH,
        production_path=PRODUCTION_PATH,
        target_col=TARGET_COLUMN,
        drift_threshold=0.3,
    )

    print("Drift ratio:", report["drift_ratio"])
    print("Overall drift detected:", report["overall_drift"])

    print("\nDrifted features:")
    for col, info in report["details"].items():
        if info["drifted"]:
            print(f"- {col} ({info['type']}), p-value={info['p_value']:.5f}")
