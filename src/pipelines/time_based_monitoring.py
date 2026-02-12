import json
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

from src.monitoring.performance import evaluate_performance
from src.monitoring.evaluator import evaluate_rules
from src.monitoring.data_drift import evaluate_data_drift
from src.monitoring.feature_drift import evaluate_feature_drift

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-monitoring")

MODEL_URI = "models:/bank-marketing@champion"
REFERENCE_PATH = "data/reference.csv"
BATCH_DIR = Path("data/batches")



def main():
    print("Starting time-based monitoring...")
    print("Batch directory exists:", BATCH_DIR.exists())

    batch_files = sorted(BATCH_DIR.glob("batch_*.csv"))
    print("Batch files found:", len(batch_files))

    if not batch_files:
        raise RuntimeError(
            f"No batch files found in {BATCH_DIR.resolve()}"
        )

    model = mlflow.sklearn.load_model(MODEL_URI)
    ref_df = pd.read_csv(REFERENCE_PATH)

    for batch_file in batch_files:
        print("\nProcessing batch:", batch_file.name)
        prod_df = pd.read_csv(batch_file)

        # ---------- Drift checks ----------
        cov_status, cov_metrics = evaluate_data_drift(ref_df, prod_df)
        feat_status, feat_metrics = evaluate_feature_drift(ref_df, prod_df)

        # ---------- Performance ----------
        ref_metrics = evaluate_performance(model, ref_df)
        prod_metrics = evaluate_performance(model, prod_df)
        perf_status, rule_results = evaluate_rules(ref_metrics, prod_metrics)

        # ---------- Final gate ----------
        final_status = "FAIL" if (
            cov_status == "FAIL"
            or feat_status == "FAIL"
            or perf_status == "FAIL"
        ) else "PASS"

        report = {
            "batch": batch_file.stem,
            "final_status": final_status,
            "covariate_drift_status": cov_status,
            "feature_drift_status": feat_status,
            "performance_status": perf_status,
            "covariate_drift": cov_metrics,
            "feature_drift": feat_metrics,
            "performance": {
                "reference": ref_metrics,
                "production": prod_metrics,
                "rules": rule_results
            }
        }

        # ---------- MLflow logging ----------
        with mlflow.start_run(run_name=batch_file.stem):
            print("MLflow run started for:", batch_file.stem)

            mlflow.log_param("batch", batch_file.stem)
            mlflow.log_param("final_status", final_status)
            mlflow.log_param("covariate_drift_status", cov_status)
            mlflow.log_param("feature_drift_status", feat_status)
            mlflow.log_param("performance_status", perf_status)

            # Performance metrics
            for k, v in prod_metrics.items():
                mlflow.log_metric(f"prod_{k}", float(v))

            # Covariate drift metrics
            for k, v in cov_metrics.items():
                mlflow.log_metric(f"covariate_{k}", float(v))

            # Feature drift metrics
            for feature, values in feat_metrics.items():
                mlflow.log_metric(
                    f"feature_drift_{feature}",
                    float(values["score"])
                )

            mlflow.log_dict(report, "monitoring_report.json")

        print(f"{batch_file.stem} → {final_status}")

    print("\nTime-based monitoring completed successfully.")


if __name__ == "__main__":
    main()
